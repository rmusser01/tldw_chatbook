"""Immutable Persona Visual graph persistence in the migrated ChaChaNotes DB."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError

from .contracts import PersonaVisualManifest, PersonaVisualManifestError
from .validation import validate_persona_visual_manifest


@dataclass(frozen=True, slots=True)
class PersonaVisualIdentity:
    """Complete optimistic identity for one active Persona Visual graph."""

    persona_id: str
    persona_revision: int
    binding_id: int
    binding_version: int
    pack_id: int
    pack_revision: int
    pack_version_id: int
    version_number: int
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class PersonaVisualPackRecord:
    """Path-free active pack metadata."""

    id: int
    title: str
    description: str
    status: str
    source_kind: str
    created_at: str
    updated_at: str
    revision: int


@dataclass(frozen=True, slots=True)
class PersonaVisualVersionRecord:
    """One immutable, validated manifest version."""

    id: int
    pack_id: int
    version_number: int
    renderer_type: str
    manifest_version: int
    manifest: PersonaVisualManifest
    manifest_sha256: str
    created_at: str


@dataclass(frozen=True, slots=True)
class PersonaVisualAssetRecord:
    """One immutable version-bound raster asset."""

    id: int
    pack_id: int
    pack_version_id: int
    asset_key: str
    role: str
    mime_type: str
    byte_count: int
    sha256: str
    width: int
    height: int
    frame_count: int | None
    duration_ms: int | None
    created_at: str


@dataclass(frozen=True, slots=True)
class PersonaVisualBindingRecord:
    """One active local-Persona binding snapshot."""

    id: int
    persona_id: str
    persona_revision: int
    pack_id: int
    active_version_id: int
    status: str
    created_at: str
    updated_at: str
    revision: int


@dataclass(frozen=True, slots=True)
class PersonaVisualGraph:
    """Complete immutable active graph returned to runtime consumers."""

    identity: PersonaVisualIdentity
    pack: PersonaVisualPackRecord
    version: PersonaVisualVersionRecord
    binding: PersonaVisualBindingRecord
    assets: tuple[PersonaVisualAssetRecord, ...]


@dataclass(frozen=True, slots=True)
class _AssetWrite:
    asset_key: str
    role: str
    storage_relpath: str
    mime_type: str
    byte_count: int
    sha256: str
    width: int
    height: int
    frame_count: int | None
    duration_ms: int | None


class PersonaVisualRepository:
    """Read and atomically write Persona Visual graphs; migrations own schema."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self.db = db

    def get_active_persona_pack(self, persona_id: str) -> PersonaVisualGraph | None:
        """Return the Persona's active graph, ignoring inactive bindings/packs."""

        _validate_persona_id(persona_id)
        try:
            with self.db.transaction():
                return self._get_active_persona_pack(persona_id)
        except CharactersRAGDBError:
            raise ValueError("persona_visual_repository_read_failed") from None

    def activate_new_pack(
        self,
        *,
        persona_id: str,
        title: str,
        manifest: object,
        manifest_storage_relpath: str,
        assets: Sequence[Mapping[str, Any]],
        expected_persona_revision: int,
        authority_guard: Callable[[], bool],
        description: str = "",
        source_kind: str = "manual",
        source_context: object | None = None,
    ) -> PersonaVisualGraph:
        """Create and activate a first immutable graph for a local Persona."""

        _validate_persona_id(persona_id)
        _validate_revision(expected_persona_revision)
        if not isinstance(title, str) or not title or not isinstance(description, str):
            raise ValueError("persona_visual_pack_invalid")
        if not isinstance(source_kind, str) or not source_kind:
            raise ValueError("persona_visual_pack_invalid")
        context_json = _source_context_json(
            {} if source_context is None else source_context
        )
        manifest_relpath = _storage_relpath(manifest_storage_relpath)
        asset_writes = _asset_writes(assets)
        manifest_json, validated_manifest, manifest_sha256 = _manifest_json(
            manifest, asset_writes
        )
        _validate_guard(authority_guard)
        self._require_owned_write_transaction()

        try:
            with self.db.transaction(immediate=True):
                if self._active_binding_row(persona_id) is not None:
                    raise ValueError("persona_visual_binding_changed")
                pack_id = int(
                    self.db.execute_query(
                        """
                        INSERT INTO persona_visual_packs(
                            title, description, status, active_version_id,
                            source_kind, source_context_json
                        ) VALUES (?, ?, 'active', NULL, ?, ?)
                        """,
                        (title, description, source_kind, context_json),
                        redact_params=True,
                    ).lastrowid
                )
                version_id = self._insert_version(
                    pack_id=pack_id,
                    version_number=1,
                    manifest_json=manifest_json,
                    manifest=validated_manifest,
                    manifest_sha256=manifest_sha256,
                    storage_relpath=manifest_relpath,
                )
                self._insert_assets(pack_id, version_id, asset_writes)
                _run_authority_guard(authority_guard)
                activated = self.db.execute_query(
                    """
                    UPDATE persona_visual_packs
                       SET active_version_id = ?, updated_at = CURRENT_TIMESTAMP
                     WHERE id = ? AND active_version_id IS NULL AND version = 1
                    """,
                    (version_id, pack_id),
                )
                if activated.rowcount != 1:
                    raise ValueError("persona_visual_identity_changed")
                self.db.execute_query(
                    """
                    INSERT INTO persona_visual_bindings(
                        persona_id, persona_revision, pack_id, active_version_id
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (persona_id, expected_persona_revision, pack_id, version_id),
                )
                graph = self._get_active_persona_pack(persona_id)
                if graph is None:
                    raise ValueError("persona_visual_activation_failed")
                return graph
        except CharactersRAGDBError:
            raise ValueError("persona_visual_repository_write_failed") from None

    def publish_version(
        self,
        *,
        persona_id: str,
        manifest: object,
        manifest_storage_relpath: str,
        assets: Sequence[Mapping[str, Any]],
        expected_identity: PersonaVisualIdentity,
        expected_persona_revision: int,
        authority_guard: Callable[[], bool],
    ) -> PersonaVisualGraph:
        """Publish and activate the next immutable version under full-graph CAS."""

        _validate_persona_id(persona_id)
        _validate_revision(expected_persona_revision)
        if not isinstance(expected_identity, PersonaVisualIdentity):
            raise ValueError("persona_visual_identity_changed")
        manifest_relpath = _storage_relpath(manifest_storage_relpath)
        asset_writes = _asset_writes(assets)
        manifest_json, validated_manifest, manifest_sha256 = _manifest_json(
            manifest, asset_writes
        )
        _validate_guard(authority_guard)
        self._require_owned_write_transaction()

        try:
            with self.db.transaction(immediate=True):
                current = self._get_active_persona_pack(persona_id)
                if current is None:
                    raise ValueError("persona_visual_identity_changed")
                if current.identity.persona_revision != expected_persona_revision:
                    raise ValueError("persona_visual_persona_revision_changed")
                if current.identity != expected_identity:
                    raise ValueError("persona_visual_identity_changed")

                source_manifest_json = self._reserve_identity(current.identity)
                next_number = int(
                    self.db.execute_query(
                        """
                        SELECT COALESCE(MAX(version_number), 0) + 1
                          FROM persona_visual_pack_versions
                         WHERE pack_id = ?
                        """,
                        (current.pack.id,),
                    ).fetchone()[0]
                )
                version_id = self._insert_version(
                    pack_id=current.pack.id,
                    version_number=next_number,
                    manifest_json=manifest_json,
                    manifest=validated_manifest,
                    manifest_sha256=manifest_sha256,
                    storage_relpath=manifest_relpath,
                )
                self._insert_assets(current.pack.id, version_id, asset_writes)
                _run_authority_guard(authority_guard)

                pack_update = self.db.execute_query(
                    """
                    UPDATE persona_visual_packs
                       SET active_version_id = ?,
                           updated_at = CURRENT_TIMESTAMP,
                           version = version + 1
                     WHERE id = ? AND status = 'active'
                       AND active_version_id = ? AND version = ?
                       AND EXISTS (
                           SELECT 1
                             FROM persona_visual_pack_versions AS source_version
                            WHERE source_version.id = ?
                              AND source_version.pack_id = ?
                              AND source_version.version_number = ?
                              AND source_version.renderer_type = ?
                              AND source_version.manifest_version = ?
                              AND source_version.manifest_json = ?
                              AND source_version.manifest_sha256 = ?
                       )
                    """,
                    (
                        version_id,
                        current.pack.id,
                        current.version.id,
                        current.pack.revision,
                        expected_identity.pack_version_id,
                        expected_identity.pack_id,
                        expected_identity.version_number,
                        current.version.renderer_type,
                        current.version.manifest_version,
                        source_manifest_json,
                        expected_identity.manifest_sha256,
                    ),
                    redact_params=True,
                )
                binding_update = self.db.execute_query(
                    """
                    UPDATE persona_visual_bindings
                       SET active_version_id = ?,
                           updated_at = CURRENT_TIMESTAMP,
                           version = version + 1
                     WHERE id = ? AND persona_id = ? AND persona_revision = ?
                       AND pack_id = ? AND active_version_id = ?
                       AND status = 'active' AND version = ?
                    """,
                    (
                        version_id,
                        current.binding.id,
                        persona_id,
                        expected_persona_revision,
                        current.pack.id,
                        current.version.id,
                        current.binding.revision,
                    ),
                )
                if pack_update.rowcount != 1 or binding_update.rowcount != 1:
                    raise ValueError("persona_visual_identity_changed")
                graph = self._get_active_persona_pack(persona_id)
                if graph is None:
                    raise ValueError("persona_visual_activation_failed")
                return graph
        except CharactersRAGDBError:
            raise ValueError("persona_visual_repository_write_failed") from None

    def archive_binding(
        self,
        *,
        persona_id: str,
        expected_identity: PersonaVisualIdentity,
    ) -> None:
        """Archive an active binding after comparing its complete identity."""

        _validate_persona_id(persona_id)
        if not isinstance(expected_identity, PersonaVisualIdentity):
            raise ValueError("persona_visual_identity_changed")
        self._require_owned_write_transaction()
        try:
            with self.db.transaction(immediate=True):
                current = self._get_active_persona_pack(persona_id)
                if current is None or current.identity != expected_identity:
                    raise ValueError("persona_visual_identity_changed")
                changed = self.db.execute_query(
                    """
                    UPDATE persona_visual_bindings
                       SET status = 'archived',
                           updated_at = CURRENT_TIMESTAMP,
                           version = version + 1
                     WHERE id = ? AND persona_id = ? AND status = 'active'
                       AND version = ? AND active_version_id = ?
                    """,
                    (
                        current.binding.id,
                        persona_id,
                        current.binding.revision,
                        current.version.id,
                    ),
                )
                if changed.rowcount != 1:
                    raise ValueError("persona_visual_identity_changed")
        except CharactersRAGDBError:
            raise ValueError("persona_visual_repository_write_failed") from None

    def _get_active_persona_pack(self, persona_id: str) -> PersonaVisualGraph | None:
        binding_row = self._active_binding_row(persona_id)
        if binding_row is None:
            return None
        binding = dict(binding_row)
        pack_row = self.db.execute_query(
            "SELECT * FROM persona_visual_packs WHERE id = ?",
            (binding["pack_id"],),
        ).fetchone()
        if pack_row is None:
            raise ValueError("persona_visual_pack_relationship_invalid")
        pack = dict(pack_row)
        if pack["status"] != "active":
            return None
        version_row = self.db.execute_query(
            "SELECT * FROM persona_visual_pack_versions WHERE id = ?",
            (binding["active_version_id"],),
        ).fetchone()
        if version_row is None:
            raise ValueError("persona_visual_pack_relationship_invalid")
        version = dict(version_row)
        if not (
            int(binding["pack_id"]) == int(pack["id"]) == int(version["pack_id"])
            and int(binding["active_version_id"])
            == int(pack["active_version_id"])
            == int(version["id"])
        ):
            raise ValueError("persona_visual_pack_relationship_invalid")

        asset_rows = self.db.execute_query(
            """
            SELECT *
              FROM persona_visual_assets
             WHERE pack_version_id = ?
             ORDER BY asset_key, id
            """,
            (version["id"],),
        ).fetchall()
        if any(
            int(row["pack_id"]) != int(pack["id"])
            or int(row["pack_version_id"]) != int(version["id"])
            for row in asset_rows
        ):
            raise ValueError("persona_visual_pack_relationship_invalid")
        assets = tuple(_asset_record(row) for row in asset_rows)
        known_assets = {
            asset.asset_key: (asset.width, asset.height) for asset in assets
        }
        manifest_json = str(version["manifest_json"])
        try:
            if hashlib.sha256(manifest_json.encode("utf-8")).hexdigest() != str(
                version["manifest_sha256"]
            ):
                raise PersonaVisualManifestError()
            manifest = validate_persona_visual_manifest(manifest_json, known_assets)
            if (
                manifest.renderer_type != version["renderer_type"]
                or manifest.manifest_version != version["manifest_version"]
            ):
                raise PersonaVisualManifestError()
        except PersonaVisualManifestError:
            raise ValueError("persona_visual_manifest_invalid") from None
        _validate_stored_source_context(pack["source_context_json"])

        identity = PersonaVisualIdentity(
            persona_id=str(binding["persona_id"]),
            persona_revision=int(binding["persona_revision"]),
            binding_id=int(binding["id"]),
            binding_version=int(binding["version"]),
            pack_id=int(pack["id"]),
            pack_revision=int(pack["version"]),
            pack_version_id=int(version["id"]),
            version_number=int(version["version_number"]),
            manifest_sha256=str(version["manifest_sha256"]),
        )
        return PersonaVisualGraph(
            identity=identity,
            pack=PersonaVisualPackRecord(
                id=int(pack["id"]),
                title=str(pack["title"]),
                description=str(pack["description"]),
                status=str(pack["status"]),
                source_kind=str(pack["source_kind"]),
                created_at=str(pack["created_at"]),
                updated_at=str(pack["updated_at"]),
                revision=int(pack["version"]),
            ),
            version=PersonaVisualVersionRecord(
                id=int(version["id"]),
                pack_id=int(version["pack_id"]),
                version_number=int(version["version_number"]),
                renderer_type=str(version["renderer_type"]),
                manifest_version=int(version["manifest_version"]),
                manifest=manifest,
                manifest_sha256=str(version["manifest_sha256"]),
                created_at=str(version["created_at"]),
            ),
            binding=PersonaVisualBindingRecord(
                id=int(binding["id"]),
                persona_id=str(binding["persona_id"]),
                persona_revision=int(binding["persona_revision"]),
                pack_id=int(binding["pack_id"]),
                active_version_id=int(binding["active_version_id"]),
                status=str(binding["status"]),
                created_at=str(binding["created_at"]),
                updated_at=str(binding["updated_at"]),
                revision=int(binding["version"]),
            ),
            assets=assets,
        )

    def _active_binding_row(self, persona_id: str):
        return self.db.execute_query(
            """
            SELECT * FROM persona_visual_bindings
             WHERE persona_id = ? AND status = 'active'
            """,
            (persona_id,),
        ).fetchone()

    def _require_owned_write_transaction(self) -> None:
        managed_depth = getattr(self.db._local, "transaction_depth", 0)
        if self.db.get_connection().in_transaction or managed_depth > 0:
            raise ValueError("persona_visual_transaction_active")

    def _insert_version(
        self,
        *,
        pack_id: int,
        version_number: int,
        manifest_json: str,
        manifest: PersonaVisualManifest,
        manifest_sha256: str,
        storage_relpath: str,
    ) -> int:
        return int(
            self.db.execute_query(
                """
                INSERT INTO persona_visual_pack_versions(
                    pack_id, version_number, renderer_type, manifest_version,
                    manifest_json, manifest_sha256, storage_relpath
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pack_id,
                    version_number,
                    manifest.renderer_type,
                    manifest.manifest_version,
                    manifest_json,
                    manifest_sha256,
                    storage_relpath,
                ),
                redact_params=True,
            ).lastrowid
        )

    def _insert_assets(
        self,
        pack_id: int,
        version_id: int,
        assets: tuple[_AssetWrite, ...],
    ) -> None:
        for asset in assets:
            self.db.execute_query(
                """
                INSERT INTO persona_visual_assets(
                    pack_id, pack_version_id, asset_key, role, storage_relpath,
                    mime_type, bytes, sha256, width, height,
                    frame_count, duration_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pack_id,
                    version_id,
                    asset.asset_key,
                    asset.role,
                    asset.storage_relpath,
                    asset.mime_type,
                    asset.byte_count,
                    asset.sha256,
                    asset.width,
                    asset.height,
                    asset.frame_count,
                    asset.duration_ms,
                ),
                redact_params=True,
            )

    def _reserve_identity(self, identity: PersonaVisualIdentity) -> str:
        binding = self.db.execute_query(
            """
            UPDATE persona_visual_bindings SET id = id
             WHERE id = ? AND persona_id = ? AND persona_revision = ?
               AND pack_id = ? AND active_version_id = ?
               AND status = 'active' AND version = ?
            """,
            (
                identity.binding_id,
                identity.persona_id,
                identity.persona_revision,
                identity.pack_id,
                identity.pack_version_id,
                identity.binding_version,
            ),
        )
        pack = self.db.execute_query(
            """
            UPDATE persona_visual_packs SET id = id
             WHERE id = ? AND status = 'active'
               AND active_version_id = ? AND version = ?
            """,
            (
                identity.pack_id,
                identity.pack_version_id,
                identity.pack_revision,
            ),
        )
        version = self.db.execute_query(
            """
            SELECT manifest_json FROM persona_visual_pack_versions
             WHERE id = ? AND pack_id = ? AND version_number = ?
               AND manifest_sha256 = ?
            """,
            (
                identity.pack_version_id,
                identity.pack_id,
                identity.version_number,
                identity.manifest_sha256,
            ),
        ).fetchone()
        if binding.rowcount != 1 or pack.rowcount != 1 or version is None:
            raise ValueError("persona_visual_identity_changed")
        return str(version["manifest_json"])


def _manifest_json(
    manifest: object,
    assets: tuple[_AssetWrite, ...],
) -> tuple[str, PersonaVisualManifest, str]:
    try:
        canonical = _json_dump(manifest)
        validated = validate_persona_visual_manifest(
            canonical,
            {asset.asset_key: (asset.width, asset.height) for asset in assets},
        )
    except (TypeError, ValueError, UnicodeError, PersonaVisualManifestError):
        raise ValueError("persona_visual_manifest_invalid") from None
    return canonical, validated, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _source_context_json(value: object) -> str:
    if type(value) is not dict:
        raise ValueError("persona_visual_source_context_invalid")
    try:
        _validate_source_context_content(value)
        return _json_dump(value)
    except (TypeError, ValueError, UnicodeError):
        raise ValueError("persona_visual_source_context_invalid") from None


def _validate_stored_source_context(value: object) -> None:
    try:
        parsed = json.loads(
            str(value),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
        if type(parsed) is not dict:
            raise ValueError
        _validate_source_context_content(parsed)
        _json_dump(parsed)
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise ValueError("persona_visual_source_context_invalid") from None


def _asset_writes(assets: object) -> tuple[_AssetWrite, ...]:
    if not isinstance(assets, Sequence) or isinstance(assets, (str, bytes)):
        raise ValueError("persona_visual_asset_invalid")
    result: list[_AssetWrite] = []
    try:
        for candidate in assets:
            if not isinstance(candidate, Mapping):
                raise ValueError
            asset_key = candidate["asset_key"]
            role = candidate["role"]
            mime_type = candidate["mime_type"]
            sha256 = candidate["sha256"]
            byte_count = candidate["bytes"]
            width = candidate["width"]
            height = candidate["height"]
            frame_count = candidate.get("frame_count")
            duration_ms = candidate.get("duration_ms")
            if not all(
                isinstance(value, str) and value
                for value in (asset_key, role, mime_type)
            ):
                raise ValueError
            if not _is_sha256(sha256):
                raise ValueError
            if not all(
                _is_positive_int(value) for value in (byte_count, width, height)
            ):
                raise ValueError
            if frame_count is not None and not _is_positive_int(frame_count):
                raise ValueError
            if duration_ms is not None and not _is_positive_int(duration_ms):
                raise ValueError
            result.append(
                _AssetWrite(
                    asset_key=asset_key,
                    role=role,
                    storage_relpath=_storage_relpath(candidate["storage_relpath"]),
                    mime_type=mime_type,
                    byte_count=byte_count,
                    sha256=sha256,
                    width=width,
                    height=height,
                    frame_count=frame_count,
                    duration_ms=duration_ms,
                )
            )
    except (KeyError, TypeError, ValueError):
        raise ValueError("persona_visual_asset_invalid") from None
    return tuple(result)


def _asset_record(row: Mapping[str, Any]) -> PersonaVisualAssetRecord:
    return PersonaVisualAssetRecord(
        id=int(row["id"]),
        pack_id=int(row["pack_id"]),
        pack_version_id=int(row["pack_version_id"]),
        asset_key=str(row["asset_key"]),
        role=str(row["role"]),
        mime_type=str(row["mime_type"]),
        byte_count=int(row["bytes"]),
        sha256=str(row["sha256"]),
        width=int(row["width"]),
        height=int(row["height"]),
        frame_count=int(row["frame_count"]) if row["frame_count"] is not None else None,
        duration_ms=int(row["duration_ms"]) if row["duration_ms"] is not None else None,
        created_at=str(row["created_at"]),
    )


def _storage_relpath(value: object) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ValueError("persona_visual_storage_invalid")
    path = PurePosixPath(value)
    parts = value.split("/")
    if (
        path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in parts)
        or (len(value) > 1 and value[1] == ":")
    ):
        raise ValueError("persona_visual_storage_invalid")
    return value


def _validate_persona_id(value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("persona_visual_persona_id_invalid")


def _validate_revision(value: object) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("persona_visual_persona_revision_invalid")


def _validate_guard(value: object) -> None:
    if not callable(value):
        raise ValueError("persona_visual_authority_guard_invalid")


def _run_authority_guard(guard: Callable[[], bool]) -> None:
    try:
        valid = guard()
    except Exception:
        raise ValueError("persona_visual_authority_changed") from None
    if valid is not True:
        raise ValueError("persona_visual_authority_changed")


def _json_dump(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    payload.encode("utf-8")
    return payload


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise ValueError


def _validate_source_context_content(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or _is_private_context_key(key):
                raise ValueError
            _validate_source_context_content(item)
        return
    if isinstance(value, list):
        for item in value:
            _validate_source_context_content(item)
        return
    if isinstance(value, str) and (
        PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()
    ):
        raise ValueError


def _is_private_context_key(key: str) -> bool:
    normalized = key.casefold().replace("-", "_")
    private_tokens = ("api_key", "password", "secret", "token", "persona", "prompt")
    return any(token in normalized for token in private_tokens)


def _is_positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
