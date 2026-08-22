"""Exact local-Persona authority for Shared Visual Identity reactions."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import visual_identity as _shared


_MAX_PORTRAIT_BYTES = 25 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class LocalPersonaVisualIdentityPortrait:
    """One bounded linked Character portrait with no filesystem identity."""

    portrait_id: str
    revision: int
    content_type: str
    sha256: str
    data: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class LocalPersonaVisualIdentityAuthority:
    """Exact local Persona and linked-portrait authority for one operation."""

    source: str
    persona_id: str
    persona_revision: int
    portrait: LocalPersonaVisualIdentityPortrait | None

    @property
    def cache_identity(self) -> tuple[str, ...]:
        """Return the complete path-free authority identity.

        Returns:
            Stable source, Persona revision, and portrait identity fields.
        """

        portrait = self.portrait
        return (
            f"source={self.source}",
            f"persona_id={self.persona_id}",
            f"persona_revision={self.persona_revision}",
            f"portrait_id={portrait.portrait_id if portrait is not None else ''}",
            f"portrait_revision={portrait.revision if portrait is not None else ''}",
            f"portrait_sha256={portrait.sha256 if portrait is not None else ''}",
        )


def capture_local_persona_visual_identity(
    local_service: object,
    persona_id: str,
) -> LocalPersonaVisualIdentityAuthority | None:
    """Capture one eligible local Persona and its optional linked portrait.

    Args:
        local_service: Local Persona profile service.
        persona_id: Exact profile-local Persona identifier.

    Returns:
        Immutable current authority, or ``None`` when the Persona is ineligible.
    """

    if type(persona_id) is not str or not persona_id or len(persona_id) > 200:
        return None
    getter = getattr(local_service, "get_persona_profile", None)
    if not callable(getter):
        return None
    try:
        record = getter(persona_id)
    except Exception:
        return None
    if type(record) is not dict:
        return None
    revision = record.get("version")
    if (
        record.get("backend") != "local"
        or type(record.get("id")) is not str
        or record.get("id") != persona_id
        or type(revision) is not int
        or revision < 1
        or record.get("deleted", False) is not False
        or record.get("is_active", True) is not True
    ):
        return None
    return LocalPersonaVisualIdentityAuthority(
        source="local",
        persona_id=persona_id,
        persona_revision=revision,
        portrait=_linked_portrait(local_service, record),
    )


def local_persona_visual_identity_is_current(
    local_service: object,
    authority: LocalPersonaVisualIdentityAuthority,
) -> bool:
    """Return whether the complete local Persona authority is unchanged.

    Args:
        local_service: Local Persona profile service.
        authority: Previously captured authority to revalidate.

    Returns:
        ``True`` only when every authority field is still current.
    """

    if type(authority) is not LocalPersonaVisualIdentityAuthority:
        return False
    return (
        capture_local_persona_visual_identity(local_service, authority.persona_id)
        == authority
    )


def resolve_persona_visual_identity(
    db: Any,
    local_service: object,
    *,
    persona_id: str,
    requested_state: str,
    manual_expression_key: str | None = None,
    user_data_dir: str | Path | None = None,
) -> _shared.VisualIdentityResolution:
    """Resolve one eligible local Persona reaction and linked portrait fallback.

    Args:
        db: Initialized profile-local character database.
        local_service: Local Persona profile service.
        persona_id: Exact profile-local Persona identifier.
        requested_state: Current Console operational state.
        manual_expression_key: Optional session-local reaction override.
        user_data_dir: Injectable profile root for profile-owned pack assets.

    Returns:
        Frozen reaction resolution with selected image bytes or stable fallback.
    """

    requested_state_key = (
        requested_state.strip().lower() if type(requested_state) is str else ""
    )
    requested_key = _shared._OPERATIONAL_EXPRESSION_KEYS.get(requested_state_key)
    manual_key = (
        _shared.normalize_expression_key(manual_expression_key)
        if manual_expression_key is not None
        else None
    )
    authority = capture_local_persona_visual_identity(local_service, persona_id)
    if authority is None:
        return _shared._placeholder_resolution(
            "persona", str(persona_id), requested_key, manual_key, "actor_unavailable"
        )

    with db.transaction():
        rows = db.execute_query(
            """
        /* visual_identity_resolver */
        WITH active_graph AS MATERIALIZED (
            SELECT b.id AS binding_id,
                   b.version AS binding_version,
                   p.id AS pack_id,
                   p.version AS pack_revision,
                   p.source_kind AS pack_source_kind,
                   v.id AS pack_version_id,
                   v.version_number AS pack_version_number,
                   v.manifest_json AS manifest_json,
                   v.default_expression_key AS version_default_expression_key
              FROM visual_identity_bindings b
              JOIN visual_identity_packs p
                ON p.id = b.pack_id
               AND p.owner_user_id = 0
               AND p.status = 'active'
               AND p.active_version_id = b.active_version_id
              JOIN visual_identity_pack_versions v
                ON v.id = b.active_version_id
               AND v.pack_id = p.id
               AND v.owner_user_id = 0
             WHERE b.owner_user_id = 0
               AND b.actor_kind = 'persona'
               AND b.actor_id = ?
               AND b.status = 'active'
        ),
        candidate_slots(expression_key, priority) AS MATERIALIZED (
            SELECT column1, column2
              FROM (VALUES (?, 0), (?, 1))
             WHERE column1 IS NOT NULL
            UNION ALL
            SELECT version_default_expression_key, 2
              FROM active_graph
             WHERE pack_version_id IS NOT NULL
               AND version_default_expression_key IS NOT NULL
            UNION ALL
            SELECT 'neutral', 3
              FROM active_graph
             WHERE pack_version_id IS NOT NULL
        ),
        deduplicated_slots AS MATERIALIZED (
            SELECT expression_key, MIN(priority) AS priority
              FROM candidate_slots
             GROUP BY expression_key
        ),
        selected_slots AS MATERIALIZED (
            SELECT slots.expression_key,
                   slots.priority,
                   (
                       SELECT a2.id
                         FROM visual_identity_assets a2
                        WHERE a2.pack_id = graph.pack_id
                          AND a2.pack_version_id = graph.pack_version_id
                          AND a2.owner_user_id = 0
                          AND a2.deleted = 0
                          AND a2.expression_key = slots.expression_key
                        ORDER BY a2.id
                        LIMIT 1
                   ) AS asset_id
              FROM deduplicated_slots slots
              CROSS JOIN active_graph graph
        )
        SELECT graph.binding_id,
               graph.binding_version,
               graph.pack_id,
               graph.pack_revision,
               graph.pack_source_kind,
               graph.pack_version_id,
               graph.pack_version_number,
               graph.manifest_json,
               graph.version_default_expression_key,
               a.id AS asset_id,
               a.expression_key AS asset_expression_key,
               a.original_expression_key AS asset_original_expression_key,
               a.display_label AS asset_display_label,
               a.storage_relpath AS asset_storage_relpath,
               a.content_type AS asset_content_type,
               a.bytes AS asset_bytes,
               a.sha256 AS asset_sha256,
               a.width AS asset_width,
               a.height AS asset_height,
               a.is_animated AS asset_is_animated,
               a.frame_count AS asset_frame_count,
               a.duration_ms AS asset_duration_ms
          FROM active_graph graph
          LEFT JOIN selected_slots selected ON 1 = 1
          LEFT JOIN visual_identity_assets a ON a.id = selected.asset_id
         ORDER BY CASE
                    WHEN selected.priority IS NOT NULL THEN selected.priority
                    ELSE 4
                  END,
                  a.id
         LIMIT 4
        """,
            (authority.persona_id, manual_key, requested_key),
        ).fetchall()
    candidates = [dict(row) for row in rows]
    for candidate in candidates:
        if candidate["asset_id"] is None:
            continue
        try:
            asset = _shared._resolution_manifest_asset(candidate)
            loaded = _shared.load_visual_identity_asset(
                asset,
                source_kind=str(candidate["pack_source_kind"]),
                user_data_dir=user_data_dir,
            )
            _shared._validate_image_bytes(loaded, decoded_pixels_before=0)
        except (TypeError, ValueError, OverflowError):
            continue
        if not local_persona_visual_identity_is_current(local_service, authority):
            return _shared._placeholder_resolution(
                "persona",
                authority.persona_id,
                requested_key,
                manual_key,
                "actor_unavailable",
            )
        source, reason = _shared._pack_resolution_category(
            str(candidate["asset_expression_key"]),
            manual_key=manual_key,
            requested_key=requested_key,
            default_key=str(candidate["version_default_expression_key"]),
        )
        manifest_sha256 = hashlib.sha256(
            str(candidate["manifest_json"]).encode("utf-8")
        ).hexdigest()
        return _shared.VisualIdentityResolution(
            actor_kind="persona",
            actor_id=authority.persona_id,
            requested_expression_key=requested_key,
            manual_expression_key=manual_key,
            resolved_expression_key=str(candidate["asset_expression_key"]),
            pack_id=int(candidate["pack_id"]),
            pack_version_id=int(candidate["pack_version_id"]),
            asset_id=int(candidate["asset_id"]),
            expression_id=None,
            storage_source=str(candidate["pack_source_kind"]),
            storage_relpath=str(candidate["asset_storage_relpath"]),
            content_type=str(candidate["asset_content_type"]),
            is_animated=bool(candidate["asset_is_animated"]),
            resolution_source=source,
            fallback_reason=reason,
            cache_identity=_persona_cache_identity(
                authority,
                requested_key=requested_key,
                manual_key=manual_key,
                resolution_source=source,
                binding_id=int(candidate["binding_id"]),
                binding_version=int(candidate["binding_version"]),
                pack_id=int(candidate["pack_id"]),
                pack_revision=int(candidate["pack_revision"]),
                pack_version_id=int(candidate["pack_version_id"]),
                pack_version_number=int(candidate["pack_version_number"]),
                manifest_sha256=manifest_sha256,
                asset_id=int(candidate["asset_id"]),
                asset_sha256=str(candidate["asset_sha256"]),
            ),
            image_bytes=loaded.data,
        )

    if not local_persona_visual_identity_is_current(local_service, authority):
        return _shared._placeholder_resolution(
            "persona",
            authority.persona_id,
            requested_key,
            manual_key,
            "actor_unavailable",
        )
    validated_portrait = _shared._validate_fallback_image(
        authority.portrait.data if authority.portrait is not None else None
    )
    if not local_persona_visual_identity_is_current(local_service, authority):
        return _shared._placeholder_resolution(
            "persona",
            authority.persona_id,
            requested_key,
            manual_key,
            "actor_unavailable",
        )
    if validated_portrait is None:
        return _shared._placeholder_resolution(
            "persona",
            authority.persona_id,
            requested_key,
            manual_key,
            "portrait_unavailable",
        )
    data, content_type, is_animated = validated_portrait
    return _shared.VisualIdentityResolution(
        actor_kind="persona",
        actor_id=authority.persona_id,
        requested_expression_key=requested_key,
        manual_expression_key=manual_key,
        resolved_expression_key=requested_key,
        pack_id=None,
        pack_version_id=None,
        asset_id=None,
        expression_id=None,
        storage_source="database",
        storage_relpath=None,
        content_type=content_type,
        is_animated=is_animated,
        resolution_source="persona_portrait",
        fallback_reason="pack_assets_unavailable" if candidates else "pack_unavailable",
        cache_identity=_persona_cache_identity(
            authority,
            requested_key=requested_key,
            manual_key=manual_key,
            resolution_source="persona_portrait",
        ),
        image_bytes=data,
    )


def _persona_cache_identity(
    authority: LocalPersonaVisualIdentityAuthority,
    *,
    requested_key: str | None,
    manual_key: str | None,
    resolution_source: str,
    binding_id: int | None = None,
    binding_version: int | None = None,
    pack_id: int | None = None,
    pack_revision: int | None = None,
    pack_version_id: int | None = None,
    pack_version_number: int | None = None,
    manifest_sha256: str | None = None,
    asset_id: int | None = None,
    asset_sha256: str | None = None,
) -> tuple[str, ...]:
    portrait = authority.portrait
    return _shared._resolution_cache_identity(
        "persona",
        authority.persona_id,
        requested_key,
        manual_key,
        resolution_source,
        "actor_source=local",
        f"persona_revision={authority.persona_revision}",
        f"portrait_id={portrait.portrait_id if portrait is not None else ''}",
        f"portrait_revision={portrait.revision if portrait is not None else ''}",
        f"portrait_sha256={portrait.sha256 if portrait is not None else ''}",
        f"binding_id={binding_id if binding_id is not None else ''}",
        f"binding_version={binding_version if binding_version is not None else ''}",
        f"pack_id={pack_id if pack_id is not None else ''}",
        f"pack_revision={pack_revision if pack_revision is not None else ''}",
        f"pack_version_id={pack_version_id if pack_version_id is not None else ''}",
        f"pack_version_number={pack_version_number if pack_version_number is not None else ''}",
        f"manifest_sha256={manifest_sha256 or ''}",
        f"asset_id={asset_id if asset_id is not None else ''}",
        f"sha256={asset_sha256 or ''}",
    )


def _linked_portrait(
    local_service: object,
    persona_record: Mapping[str, Any],
) -> LocalPersonaVisualIdentityPortrait | None:
    character_id = persona_record.get("character_card_id")
    getter = getattr(local_service, "get_character", None)
    if type(character_id) is not int or character_id < 1 or not callable(getter):
        return None
    try:
        character = getter(character_id)
    except Exception:
        return None
    if type(character) is not dict:
        return None
    revision = character.get("version")
    data = character.get("image")
    if (
        type(character.get("id")) is not int
        or character.get("id") != character_id
        or type(revision) is not int
        or revision < 0
        or character.get("deleted", False) is not False
        or type(data) is not bytes
        or not data
        or len(data) > _MAX_PORTRAIT_BYTES
    ):
        return None
    content_type = _portrait_content_type(data)
    if content_type is None:
        return None
    return LocalPersonaVisualIdentityPortrait(
        portrait_id=f"local-character:{character_id}",
        revision=revision,
        content_type=content_type,
        sha256=hashlib.sha256(data).hexdigest(),
        data=data,
    )


def _portrait_content_type(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return None


__all__ = [
    "LocalPersonaVisualIdentityAuthority",
    "LocalPersonaVisualIdentityPortrait",
    "capture_local_persona_visual_identity",
    "local_persona_visual_identity_is_current",
    "resolve_persona_visual_identity",
]
