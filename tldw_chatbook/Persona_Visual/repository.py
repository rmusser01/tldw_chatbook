"""Immutable Persona Visual graph persistence in the migrated ChaChaNotes DB."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError

from .contracts import (
    ALLOWED_ASSET_MIME_TYPES,
    ALLOWED_ASSET_ROLES,
    MAX_ASSET_DIMENSION,
    MAX_FRAME_DURATION_MS,
    MAX_FRAMES_PER_ANIMATION,
    PersonaVisualManifest,
    PersonaVisualManifestError,
)
from .validation import validate_persona_visual_manifest


_SOURCE_CONTEXT_KEYS = frozenset(
    {"source_id", "provenance", "license", "source_server_commit"}
)
_MAX_SOURCE_CONTEXT_VALUE_LENGTH = 256
_ASSET_KEY_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_SOURCE_KIND_PATTERN = re.compile(r"[a-z][a-z0-9_.:-]{0,63}\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_GRAPH_STATUSES = frozenset({"active", "archived", "deleted"})
_SQLITE_CORRUPTION_CODES = frozenset(
    {sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_FORMAT, sqlite3.SQLITE_NOTADB}
)
_SQLITE_UTF8_DECODE_PREFIX = "Could not decode to UTF-8 column "


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
class AssetWrite:
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
        except (UnicodeError, RecursionError, TypeError, OverflowError):
            raise ValueError("persona_visual_graph_invalid") from None
        except (sqlite3.Error, CharactersRAGDBError):
            raise ValueError("persona_visual_repository_read_failed") from None

    def _get_active_asset_storage_key(
        self,
        identity: PersonaVisualIdentity,
        asset: PersonaVisualAssetRecord,
    ) -> str:
        """Resolve private storage only under an exact active immutable snapshot."""

        if (
            type(identity) is not PersonaVisualIdentity
            or type(asset) is not PersonaVisualAssetRecord
        ):
            raise ValueError("persona_visual_asset_storage_unavailable")
        try:
            with self.db.transaction():
                row = _fetchone(
                    self.db.execute_query(
                        """
                        SELECT asset.storage_relpath
                          FROM persona_visual_bindings AS binding
                          JOIN persona_visual_packs AS pack
                            ON pack.id = binding.pack_id
                          JOIN persona_visual_pack_versions AS version_row
                            ON version_row.id = binding.active_version_id
                           AND version_row.pack_id = binding.pack_id
                          JOIN persona_visual_assets AS asset
                            ON asset.pack_version_id = version_row.id
                           AND asset.pack_id = pack.id
                         WHERE binding.id = ? AND binding.persona_id = ?
                           AND binding.persona_revision = ?
                           AND binding.version = ? AND binding.status = 'active'
                           AND binding.pack_id = ?
                           AND binding.active_version_id = ?
                           AND pack.id = ? AND pack.version = ?
                           AND pack.status = 'active'
                           AND pack.active_version_id = ?
                           AND version_row.id = ?
                           AND version_row.version_number = ?
                           AND version_row.manifest_sha256 = ?
                           AND asset.id = ? AND asset.pack_id = ?
                           AND asset.pack_version_id = ?
                           AND asset.asset_key = ? AND asset.role = ?
                           AND asset.mime_type = ? AND asset.bytes = ?
                           AND asset.sha256 = ? AND asset.width = ?
                           AND asset.height = ? AND asset.frame_count IS ?
                           AND asset.duration_ms IS ? AND asset.created_at = ?
                        """,
                        (
                            identity.binding_id,
                            identity.persona_id,
                            identity.persona_revision,
                            identity.binding_version,
                            identity.pack_id,
                            identity.pack_version_id,
                            identity.pack_id,
                            identity.pack_revision,
                            identity.pack_version_id,
                            identity.pack_version_id,
                            identity.version_number,
                            identity.manifest_sha256,
                            asset.id,
                            asset.pack_id,
                            asset.pack_version_id,
                            asset.asset_key,
                            asset.role,
                            asset.mime_type,
                            asset.byte_count,
                            asset.sha256,
                            asset.width,
                            asset.height,
                            asset.frame_count,
                            asset.duration_ms,
                            asset.created_at,
                        ),
                        redact_params=True,
                    )
                )
                if row is None:
                    raise ValueError("persona_visual_asset_storage_unavailable")
                return _decode_storage_relpath(row["storage_relpath"])
        except ValueError:
            raise ValueError("persona_visual_asset_storage_unavailable") from None
        except (sqlite3.Error, CharactersRAGDBError):
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
        try:
            _input_text(title, 256)
            _input_text(description, 4096, allow_empty=True)
            source_kind = _input_text(source_kind, 64)
        except (TypeError, ValueError, UnicodeError):
            raise ValueError("persona_visual_pack_invalid")
        if _SOURCE_KIND_PATTERN.fullmatch(source_kind) is None:
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
                transaction_connection = self.db.get_connection()
                if self._active_binding_record(persona_id) is not None:
                    raise ValueError("persona_visual_binding_changed")
                _run_authority_guard(authority_guard, self.db, transaction_connection)
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
        except (UnicodeError, RecursionError, TypeError, OverflowError):
            raise ValueError("persona_visual_graph_invalid") from None
        except (sqlite3.Error, CharactersRAGDBError):
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
                transaction_connection = self.db.get_connection()
                current = self._get_active_persona_pack(persona_id)
                if current is None:
                    raise ValueError("persona_visual_identity_changed")
                if current.identity.persona_revision != expected_persona_revision:
                    raise ValueError("persona_visual_persona_revision_changed")
                if current.identity != expected_identity:
                    raise ValueError("persona_visual_identity_changed")

                source_manifest_json = self._read_identity_snapshot(current.identity)
                next_number = _db_positive_int(
                    _fetchone(
                        self.db.execute_query(
                            """
                        SELECT COALESCE(MAX(version_number), 0) + 1
                          FROM persona_visual_pack_versions
                         WHERE pack_id = ?
                        """,
                            (current.pack.id,),
                        )
                    )[0]
                )
                _run_authority_guard(authority_guard, self.db, transaction_connection)
                version_id = self._insert_version(
                    pack_id=current.pack.id,
                    version_number=next_number,
                    manifest_json=manifest_json,
                    manifest=validated_manifest,
                    manifest_sha256=manifest_sha256,
                    storage_relpath=manifest_relpath,
                )
                self._insert_assets(current.pack.id, version_id, asset_writes)

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
        except (UnicodeError, RecursionError, TypeError, OverflowError):
            raise ValueError("persona_visual_graph_invalid") from None
        except (sqlite3.Error, CharactersRAGDBError):
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
        except (UnicodeError, RecursionError, TypeError, OverflowError):
            raise ValueError("persona_visual_graph_invalid") from None
        except (sqlite3.Error, CharactersRAGDBError):
            raise ValueError("persona_visual_repository_write_failed") from None

    def _get_active_persona_pack(self, persona_id: str) -> PersonaVisualGraph | None:
        binding = self._active_binding_record(persona_id)
        if binding is None:
            return None
        pack_row = _fetchone(
            self.db.execute_query(
                "SELECT * FROM persona_visual_packs WHERE id = ?",
                (binding.pack_id,),
            )
        )
        if pack_row is None:
            raise ValueError("persona_visual_pack_relationship_invalid")
        pack, pack_active_version_id = _decode_pack(pack_row)
        if pack.status != "active":
            return None
        version_row = _fetchone(
            self.db.execute_query(
                "SELECT * FROM persona_visual_pack_versions WHERE id = ?",
                (binding.active_version_id,),
            )
        )
        if version_row is None:
            raise ValueError("persona_visual_pack_relationship_invalid")

        asset_rows = _fetchall(
            self.db.execute_query(
                """
                SELECT *
                  FROM persona_visual_assets
                 WHERE pack_version_id = ?
                 ORDER BY asset_key, id
                """,
                (binding.active_version_id,),
            )
        )
        assets = tuple(_decode_asset(row) for row in asset_rows)
        version = _decode_version(version_row, assets)
        if not (
            binding.pack_id == pack.id == version.pack_id
            and binding.active_version_id == pack_active_version_id == version.id
        ):
            raise ValueError("persona_visual_pack_relationship_invalid")
        if any(
            asset.pack_id != pack.id or asset.pack_version_id != version.id
            for asset in assets
        ):
            raise ValueError("persona_visual_pack_relationship_invalid")

        identity = PersonaVisualIdentity(
            persona_id=binding.persona_id,
            persona_revision=binding.persona_revision,
            binding_id=binding.id,
            binding_version=binding.revision,
            pack_id=pack.id,
            pack_revision=pack.revision,
            pack_version_id=version.id,
            version_number=version.version_number,
            manifest_sha256=version.manifest_sha256,
        )
        return PersonaVisualGraph(
            identity=identity,
            pack=pack,
            version=version,
            binding=binding,
            assets=assets,
        )

    def _active_binding_record(
        self, persona_id: str
    ) -> PersonaVisualBindingRecord | None:
        row = _fetchone(
            self.db.execute_query(
                """
                SELECT * FROM persona_visual_bindings
                 WHERE persona_id = ? AND status = 'active'
                """,
                (persona_id,),
            )
        )
        return None if row is None else _decode_binding(row)

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
        assets: tuple[AssetWrite, ...],
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

    def _read_identity_snapshot(self, identity: PersonaVisualIdentity) -> str:
        version = _fetchone(
            self.db.execute_query(
                """
            SELECT version_row.manifest_json
              FROM persona_visual_bindings AS binding
              JOIN persona_visual_packs AS pack ON pack.id = binding.pack_id
              JOIN persona_visual_pack_versions AS version_row
                ON version_row.id = binding.active_version_id
               AND version_row.pack_id = binding.pack_id
             WHERE binding.id = ? AND binding.persona_id = ?
               AND binding.persona_revision = ? AND binding.pack_id = ?
               AND binding.active_version_id = ?
               AND binding.status = 'active' AND binding.version = ?
               AND pack.status = 'active' AND pack.active_version_id = ?
               AND pack.version = ? AND version_row.version_number = ?
               AND version_row.manifest_sha256 = ?
                """,
                (
                    identity.binding_id,
                    identity.persona_id,
                    identity.persona_revision,
                    identity.pack_id,
                    identity.pack_version_id,
                    identity.binding_version,
                    identity.pack_version_id,
                    identity.pack_revision,
                    identity.version_number,
                    identity.manifest_sha256,
                ),
            )
        )
        if version is None:
            raise ValueError("persona_visual_identity_changed")
        return str(version["manifest_json"])


def _manifest_json(
    manifest: object,
    assets: tuple[AssetWrite, ...],
) -> tuple[str, PersonaVisualManifest, str]:
    try:
        canonical = _json_dump(manifest)
        validated = validate_persona_visual_manifest(
            canonical,
            {asset.asset_key: (asset.width, asset.height) for asset in assets},
        )
    except (
        TypeError,
        ValueError,
        UnicodeError,
        RecursionError,
        PersonaVisualManifestError,
    ):
        raise ValueError("persona_visual_manifest_invalid") from None
    return canonical, validated, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _source_context_json(value: object) -> str:
    if type(value) is not dict:
        raise ValueError("persona_visual_source_context_invalid")
    try:
        _validate_source_context_content(value)
        return _json_dump(value)
    except (TypeError, ValueError, UnicodeError, RecursionError):
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


def _asset_writes(assets: object) -> tuple[AssetWrite, ...]:
    if not isinstance(assets, Sequence) or isinstance(assets, (str, bytes)):
        raise ValueError("persona_visual_asset_invalid")
    result: list[AssetWrite] = []
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
            if (
                role not in ALLOWED_ASSET_ROLES
                or mime_type not in ALLOWED_ASSET_MIME_TYPES
            ):
                raise ValueError
            asset_key = _asset_key(asset_key)
            if not _is_sha256(sha256):
                raise ValueError
            if not all(
                _is_positive_int(value) for value in (byte_count, width, height)
            ):
                raise ValueError
            if width > MAX_ASSET_DIMENSION or height > MAX_ASSET_DIMENSION:
                raise ValueError
            if frame_count is not None and (
                not _is_positive_int(frame_count)
                or frame_count > MAX_FRAMES_PER_ANIMATION
            ):
                raise ValueError
            if duration_ms is not None and (
                not _is_positive_int(duration_ms) or duration_ms > MAX_FRAME_DURATION_MS
            ):
                raise ValueError
            result.append(
                AssetWrite(
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


def _decode_pack(
    row: Mapping[str, Any],
) -> tuple[PersonaVisualPackRecord, int]:
    """Decode an active-graph pack row without coercing corrupt values."""

    try:
        source_kind = _db_text(row["source_kind"], 64)
        if _SOURCE_KIND_PATTERN.fullmatch(source_kind) is None:
            raise ValueError
        _validate_stored_source_context(row["source_context_json"])
        active_version_id = _db_positive_int(row["active_version_id"])
        return (
            PersonaVisualPackRecord(
                id=_db_positive_int(row["id"]),
                title=_db_text(row["title"], 256),
                description=_db_text(row["description"], 4096, allow_empty=True),
                status=_db_enum(row["status"], _GRAPH_STATUSES),
                source_kind=source_kind,
                created_at=_db_timestamp(row["created_at"]),
                updated_at=_db_timestamp(row["updated_at"]),
                revision=_db_positive_int(row["version"]),
            ),
            active_version_id,
        )
    except ValueError as exc:
        if str(exc) == "persona_visual_source_context_invalid":
            raise
        raise ValueError("persona_visual_graph_invalid") from None
    except (KeyError, TypeError, UnicodeError, OverflowError):
        raise ValueError("persona_visual_graph_invalid") from None


def _decode_version(
    row: Mapping[str, Any],
    assets: tuple[PersonaVisualAssetRecord, ...],
) -> PersonaVisualVersionRecord:
    """Decode and attest one immutable manifest row and its asset set."""

    try:
        renderer_type = _db_enum(row["renderer_type"], frozenset({"sprite_frames"}))
        manifest_version = _db_positive_int(row["manifest_version"])
        if manifest_version != 1:
            raise ValueError
        manifest_json = _db_text(row["manifest_json"], 2 * 1024 * 1024)
        manifest_sha256 = _db_digest(row["manifest_sha256"])
        _decode_storage_relpath(row["storage_relpath"])
        record = PersonaVisualVersionRecord(
            id=_db_positive_int(row["id"]),
            pack_id=_db_positive_int(row["pack_id"]),
            version_number=_db_positive_int(row["version_number"]),
            renderer_type=renderer_type,
            manifest_version=manifest_version,
            manifest=_validate_stored_manifest(
                manifest_json,
                manifest_sha256,
                renderer_type,
                manifest_version,
                assets,
            ),
            manifest_sha256=manifest_sha256,
            created_at=_db_timestamp(row["created_at"]),
        )
    except ValueError as exc:
        if str(exc) == "persona_visual_manifest_invalid":
            raise
        raise ValueError("persona_visual_graph_invalid") from None
    except (KeyError, TypeError, UnicodeError, OverflowError, RecursionError):
        raise ValueError("persona_visual_graph_invalid") from None
    return record


def _decode_asset(row: Mapping[str, Any]) -> PersonaVisualAssetRecord:
    """Decode one public asset record using the persisted domain limits."""

    try:
        asset_key = _asset_key(row["asset_key"])
        _decode_storage_relpath(row["storage_relpath"])
        return PersonaVisualAssetRecord(
            id=_db_positive_int(row["id"]),
            pack_id=_db_positive_int(row["pack_id"]),
            pack_version_id=_db_positive_int(row["pack_version_id"]),
            asset_key=asset_key,
            role=_db_enum(row["role"], ALLOWED_ASSET_ROLES),
            mime_type=_db_enum(row["mime_type"], frozenset(ALLOWED_ASSET_MIME_TYPES)),
            byte_count=_db_positive_int(row["bytes"]),
            sha256=_db_digest(row["sha256"]),
            width=_db_positive_int(row["width"], MAX_ASSET_DIMENSION),
            height=_db_positive_int(row["height"], MAX_ASSET_DIMENSION),
            frame_count=_db_optional_positive_int(
                row["frame_count"], MAX_FRAMES_PER_ANIMATION
            ),
            duration_ms=_db_optional_positive_int(
                row["duration_ms"], MAX_FRAME_DURATION_MS
            ),
            created_at=_db_timestamp(row["created_at"]),
        )
    except (KeyError, TypeError, ValueError, UnicodeError, OverflowError):
        raise ValueError("persona_visual_graph_invalid") from None


def _decode_binding(row: Mapping[str, Any]) -> PersonaVisualBindingRecord:
    """Decode one binding before selecting its active graph."""

    try:
        persona_id = _db_text(row["persona_id"], 200)
        return PersonaVisualBindingRecord(
            id=_db_positive_int(row["id"]),
            persona_id=persona_id,
            persona_revision=_db_nonnegative_int(row["persona_revision"]),
            pack_id=_db_positive_int(row["pack_id"]),
            active_version_id=_db_positive_int(row["active_version_id"]),
            status=_db_enum(row["status"], _GRAPH_STATUSES),
            created_at=_db_timestamp(row["created_at"]),
            updated_at=_db_timestamp(row["updated_at"]),
            revision=_db_positive_int(row["version"]),
        )
    except (KeyError, TypeError, ValueError, UnicodeError, OverflowError):
        raise ValueError("persona_visual_graph_invalid") from None


def _validate_stored_manifest(
    manifest_json: str,
    manifest_sha256: str,
    renderer_type: str,
    manifest_version: int,
    assets: tuple[PersonaVisualAssetRecord, ...],
) -> PersonaVisualManifest:
    try:
        if hashlib.sha256(manifest_json.encode("utf-8")).hexdigest() != manifest_sha256:
            raise PersonaVisualManifestError()
        manifest = validate_persona_visual_manifest(
            manifest_json,
            {asset.asset_key: (asset.width, asset.height) for asset in assets},
        )
        if (
            manifest.renderer_type != renderer_type
            or manifest.manifest_version != manifest_version
        ):
            raise PersonaVisualManifestError()
        return manifest
    except (PersonaVisualManifestError, RecursionError, UnicodeError):
        raise ValueError("persona_visual_manifest_invalid") from None


def _decode_storage_relpath(value: object) -> str:
    try:
        return _storage_relpath(value)
    except (TypeError, ValueError, UnicodeError):
        raise ValueError("persona_visual_graph_invalid") from None


def _storage_relpath(value: object) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ValueError("persona_visual_storage_invalid")
    path = PurePosixPath(value)
    parts = value.split("/")
    if (
        path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in parts)
        or any(":" in part for part in parts)
    ):
        raise ValueError("persona_visual_storage_invalid")
    return value


def _asset_key(value: object) -> str:
    if not isinstance(value, str) or _ASSET_KEY_PATTERN.fullmatch(value) is None:
        raise ValueError
    return value


def _validate_persona_id(value: object) -> None:
    try:
        _input_text(value, 200)
    except (TypeError, ValueError, UnicodeError):
        raise ValueError("persona_visual_persona_id_invalid") from None


def _validate_revision(value: object) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("persona_visual_persona_revision_invalid")


def _validate_guard(value: object) -> None:
    if not callable(value):
        raise ValueError("persona_visual_authority_guard_invalid")


def _run_authority_guard(
    guard: Callable[[], bool],
    db: CharactersRAGDB,
    transaction_connection: object,
) -> None:
    if not isinstance(transaction_connection, sqlite3.Connection):
        raise ValueError("persona_visual_authority_changed")
    connection = transaction_connection
    allowed_actions = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
    }
    denied = False

    def read_only_authorizer(
        action: int,
        _arg1: str | None,
        _arg2: str | None,
        _database: str | None,
        _trigger: str | None,
    ) -> int:
        nonlocal denied
        if action in allowed_actions:
            return sqlite3.SQLITE_OK
        denied = True
        return sqlite3.SQLITE_DENY

    valid = False
    released = False
    authorizer_installed = False
    try:
        connection.execute("SAVEPOINT persona_visual_authority_guard")
        changes_before = connection.total_changes
        connection.set_authorizer(read_only_authorizer)
        authorizer_installed = True
        try:
            valid = guard() is True
        except Exception:
            valid = False
        finally:
            connection.set_authorizer(None)
            authorizer_installed = False
        ownership_preserved = (
            db.get_connection() is connection
            and connection.in_transaction
            and getattr(db._local, "transaction_depth", 0) == 1
            and connection.total_changes == changes_before
        )
        if ownership_preserved:
            connection.execute("RELEASE SAVEPOINT persona_visual_authority_guard")
            released = True
    except (sqlite3.Error, TypeError, ValueError, OverflowError):
        valid = False
    finally:
        if authorizer_installed:
            try:
                connection.set_authorizer(None)
            except sqlite3.Error:
                pass
    if (
        not valid
        or denied
        or not released
        or db.get_connection() is not connection
        or not connection.in_transaction
        or getattr(db._local, "transaction_depth", 0) != 1
    ):
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
    if type(value) is not dict or not set(value) <= _SOURCE_CONTEXT_KEYS:
        raise ValueError
    for item in value.values():
        if not isinstance(item, str) or not item:
            raise ValueError
        item.encode("utf-8")
        stripped = item.strip()
        if (
            len(item) > _MAX_SOURCE_CONTEXT_VALUE_LENGTH
            or "/" in item
            or "\\" in item
            or any(ord(character) < 32 for character in item)
            or stripped in {".", ".."}
            or stripped.startswith(("{", "[", "~"))
        ):
            raise ValueError


def _is_positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _input_text(value: object, maximum: int, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError
    value.encode("utf-8")
    if (not allow_empty and not value) or len(value) > maximum:
        raise ValueError
    return value


def _fetchone(cursor: sqlite3.Cursor) -> Any:
    try:
        return cursor.fetchone()
    except sqlite3.Error as exc:
        if not _is_sqlite_graph_corruption(exc):
            raise
        raise ValueError("persona_visual_graph_invalid") from None
    except (UnicodeError, RecursionError, TypeError, OverflowError):
        raise ValueError("persona_visual_graph_invalid") from None


def _fetchall(cursor: sqlite3.Cursor) -> list[Any]:
    try:
        return cursor.fetchall()
    except sqlite3.Error as exc:
        if not _is_sqlite_graph_corruption(exc):
            raise
        raise ValueError("persona_visual_graph_invalid") from None
    except (UnicodeError, RecursionError, TypeError, OverflowError):
        raise ValueError("persona_visual_graph_invalid") from None


def _is_sqlite_graph_corruption(exc: sqlite3.Error) -> bool:
    error_code = getattr(exc, "sqlite_errorcode", None)
    return (
        isinstance(exc, sqlite3.OperationalError)
        and str(exc).startswith(_SQLITE_UTF8_DECODE_PREFIX)
    ) or (type(error_code) is int and error_code & 0xFF in _SQLITE_CORRUPTION_CODES)


def _db_int(value: object) -> int:
    try:
        if type(value) is not int:
            raise TypeError
        return int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("persona_visual_graph_invalid") from None


def _db_positive_int(value: object, maximum: int | None = None) -> int:
    result = _db_int(value)
    if result <= 0 or (maximum is not None and result > maximum):
        raise ValueError("persona_visual_graph_invalid")
    return result


def _db_nonnegative_int(value: object) -> int:
    result = _db_int(value)
    if result < 0:
        raise ValueError("persona_visual_graph_invalid")
    return result


def _db_optional_positive_int(value: object, maximum: int) -> int | None:
    return None if value is None else _db_positive_int(value, maximum)


def _db_text(value: object, maximum: int, *, allow_empty: bool = False) -> str:
    try:
        if not isinstance(value, str):
            raise TypeError
        value.encode("utf-8")
        if (not allow_empty and not value) or len(value) > maximum:
            raise ValueError
        return value
    except (TypeError, ValueError, UnicodeError, OverflowError):
        raise ValueError("persona_visual_graph_invalid") from None


def _db_enum(value: object, allowed: Collection[str]) -> str:
    result = _db_text(value, 64)
    if result not in allowed:
        raise ValueError("persona_visual_graph_invalid")
    return result


def _db_digest(value: object) -> str:
    result = _db_text(value, 64)
    if _SHA256_PATTERN.fullmatch(result) is None:
        raise ValueError("persona_visual_graph_invalid")
    return result


def _db_timestamp(value: object) -> str:
    result = _db_text(value, 19)
    try:
        datetime.strptime(result, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, OverflowError):
        raise ValueError("persona_visual_graph_invalid") from None
    return result


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
