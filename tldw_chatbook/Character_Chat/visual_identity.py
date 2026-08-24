"""Visual Identity expression, manifest, and immutable asset contracts."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import sqlite3
import stat
import threading
import warnings
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib import resources
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from loguru import logger
from PIL import Image, UnidentifiedImageError

from tldw_chatbook.config import get_user_data_dir
from tldw_chatbook.Utils.path_validation import validate_path
from tldw_chatbook.Utils.private_paths import secure_private_directory

# Begin pinned server normalization block (byte-for-byte from):
# tldw_Server_API/app/core/Visual_Identities/expression_slots.py at
# commit 385afa951922c8a9dc2002c675bb6cad65e4ac23.
CANONICAL_EXPRESSION_SLOTS = (
    "neutral",
    "happy",
    "excited",
    "sad",
    "angry",
    "thinking",
    "confused",
    "surprised",
)
CUSTOM_EXPRESSION_PREFIX = "custom:"

EXPRESSION_ALIASES = {
    "default": "neutral",
    "normal": "neutral",
    "calm": "neutral",
    "joy": "happy",
    "joyful": "happy",
    "cheerful": "happy",
    "hype": "excited",
    "thrilled": "excited",
    "upset": "sad",
    "sorrowful": "sad",
    "mad": "angry",
    "annoyed": "angry",
    "furious": "angry",
    "anger": "angry",
    "thoughtful": "thinking",
    "pondering": "thinking",
    "unsure": "confused",
    "puzzled": "confused",
    "shocked": "surprised",
    "astonished": "surprised",
}

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_expression_key(value: str) -> str | None:
    """Normalize a user-facing expression label into a canonical or custom key."""
    if not isinstance(value, str):
        return None

    raw_value = value.strip()
    if not raw_value:
        return None

    if raw_value.lower().startswith(CUSTOM_EXPRESSION_PREFIX):
        custom_part = raw_value[len(CUSTOM_EXPRESSION_PREFIX) :]
        custom_key = _sanitize_expression_token(custom_part)
        return f"{CUSTOM_EXPRESSION_PREFIX}{custom_key}" if custom_key else None

    normalized = _sanitize_expression_token(raw_value)
    if not normalized:
        return None
    if normalized in CANONICAL_EXPRESSION_SLOTS:
        return normalized
    alias = EXPRESSION_ALIASES.get(normalized)
    if alias is not None:
        return alias
    return f"{CUSTOM_EXPRESSION_PREFIX}{normalized}"


def normalize_expression_filename(filename: str) -> str | None:
    """Normalize a source filename stem into a canonical or custom expression key."""
    if not isinstance(filename, str):
        return None

    basename = filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
    if "." in basename:
        basename = basename.rsplit(".", 1)[0]
    normalized = _sanitize_expression_token(basename)
    if not normalized:
        return None
    if normalized in CANONICAL_EXPRESSION_SLOTS:
        return normalized
    alias = EXPRESSION_ALIASES.get(normalized)
    if alias is not None:
        return alias
    return f"{CUSTOM_EXPRESSION_PREFIX}{normalized}"


def is_custom_expression_key(value: str) -> bool:
    """Return whether a value normalizes to a custom expression key."""
    normalized = normalize_expression_key(value)
    return normalized is not None and normalized.startswith(CUSTOM_EXPRESSION_PREFIX)


def display_label_for_expression_key(value: str) -> str:
    """Build a human-readable label for a canonical, alias, or custom expression key."""
    normalized = normalize_expression_key(value)
    if normalized is None:
        return ""
    if normalized.startswith(CUSTOM_EXPRESSION_PREFIX):
        normalized = normalized[len(CUSTOM_EXPRESSION_PREFIX) :]
    return normalized.replace("_", " ").title()


def _sanitize_expression_token(value: str) -> str:
    normalized = _NON_ALNUM_RE.sub("_", value.strip().lower())
    return normalized.strip("_")


# End pinned server normalization block.

SAMIRA_REACTION_LABELS = (
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "neutral",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "thinking",
    "speaking",
    "error",
)

SAMIRA_EXPRESSION_KEYS = {
    "admiration": "custom:admiration",
    "amusement": "custom:amusement",
    "anger": "angry",
    "annoyance": "custom:annoyance",
    "approval": "custom:approval",
    "caring": "custom:caring",
    "confusion": "confused",
    "curiosity": "custom:curiosity",
    "desire": "custom:desire",
    "disappointment": "custom:disappointment",
    "disapproval": "custom:disapproval",
    "disgust": "custom:disgust",
    "embarrassment": "custom:embarrassment",
    "excitement": "excited",
    "fear": "custom:fear",
    "gratitude": "custom:gratitude",
    "grief": "custom:grief",
    "joy": "happy",
    "love": "custom:love",
    "nervousness": "custom:nervousness",
    "neutral": "neutral",
    "optimism": "custom:optimism",
    "pride": "custom:pride",
    "realization": "custom:realization",
    "relief": "custom:relief",
    "remorse": "custom:remorse",
    "sadness": "sad",
    "surprise": "surprised",
    "thinking": "thinking",
    "speaking": "custom:speaking",
    "error": "custom:error",
}

SAMIRA_PACK_ID = "tldw.builtin.samira.reactions"
SAMIRA_MANIFEST_SCHEMA_ID = "tldw.visual_identity_pack/v1"
SAMIRA_LICENSE = "AGPL-3.0-or-later"
SAMIRA_DEFAULT_EXPRESSION_KEY = "neutral"
SAMIRA_SERVER_COMMIT = "385afa951922c8a9dc2002c675bb6cad65e4ac23"

SAMIRA_MAX_REACTION_BYTES = 1024 * 1024
SAMIRA_MAX_REACTIONS_BYTES = 16 * 1024 * 1024
SAMIRA_MAX_DIRECTORY_BYTES = 20 * 1024 * 1024

MAX_EXPRESSION_ASSET_BYTES = 25 * 1024 * 1024
MAX_EXPRESSION_IMAGE_DIMENSION = 4096
MAX_EXPRESSION_FRAME_COUNT = 512
MAX_EXPRESSION_PACK_ASSETS = 128
MAX_EXPRESSION_TOTAL_BYTES = 256 * 1024 * 1024
MAX_EXPRESSION_ASSET_DECODED_PIXELS = MAX_EXPRESSION_IMAGE_DIMENSION**2 * 4
MAX_EXPRESSION_PACK_DECODED_PIXELS = MAX_EXPRESSION_IMAGE_DIMENSION**2 * 16

_READ_CHUNK_SIZE = 1024 * 1024

_LOWER_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_LICENSE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+-]*\Z")
_EXPECTED_IMAGE_FORMATS = {
    "image/gif": "GIF",
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}
_IMAGE_CONTENT_TYPES_BY_FORMAT = {
    image_format: content_type
    for content_type, image_format in _EXPECTED_IMAGE_FORMATS.items()
}
_USER_SOURCE_KINDS = frozenset({"manual"})


@dataclass(frozen=True, slots=True)
class VisualIdentityManifestAsset:
    """One immutable asset declared by a validated pack manifest."""

    expression_key: str
    original_label: str
    display_label: str
    storage_relpath: str
    content_type: str
    bytes: int
    width: int
    height: int
    sha256: str
    is_animated: bool
    frame_count: int
    duration_ms: int | None

    @property
    def relative_filename(self) -> str:
        """Return the digest-facing relative filename."""
        return self.storage_relpath


@dataclass(frozen=True, slots=True)
class VisualIdentityManifest:
    """Validated immutable subset of a Visual Identity pack manifest."""

    schema_id: str
    pack_id: str
    title: str
    license: str
    default_expression_key: str
    source_server_commit: str | None
    pack_content_sha256: str
    assets: tuple[VisualIdentityManifestAsset, ...]


@dataclass(frozen=True, slots=True)
class LoadedVisualIdentityAsset:
    """Verified bytes for one selected Visual Identity asset."""

    asset: VisualIdentityManifestAsset
    data: bytes


@dataclass(frozen=True, slots=True)
class VisualIdentityResolution:
    """One selected reaction source plus bytes ready for off-thread decoding."""

    actor_kind: str
    actor_id: str
    requested_expression_key: str | None
    manual_expression_key: str | None
    resolved_expression_key: str | None
    pack_id: int | None
    pack_version_id: int | None
    asset_id: int | None
    expression_id: int | None
    storage_source: str
    storage_relpath: str | None
    content_type: str | None
    is_animated: bool
    resolution_source: str
    fallback_reason: str
    cache_identity: tuple[str, ...]
    image_bytes: bytes | None


class VisualIdentityPublicationError(ValueError):
    """Stable publication failure with an optional profile-relative orphan."""

    def __init__(
        self, category: str, *, cleanup_candidate_relpath: str | None = None
    ) -> None:
        super().__init__(category)
        self.category = category
        self.cleanup_candidate_relpath = cleanup_candidate_relpath


@dataclass(slots=True)
class VisualIdentityCandidate:
    """In-memory edits against one immutable active pack identity."""

    actor_kind: str
    actor_id: str
    old_pack_id: int | None
    old_version_id: int | None
    old_binding_id: int | None
    old_binding_version: int | None
    old_pack_version: int | None
    source_kind: str
    title: str
    description: str
    default_expression_key: str
    original_default_expression_key: str
    source_context: dict[str, Any]
    assets: tuple[dict[str, Any], ...] = field(repr=False)
    actor_authority: tuple[str, ...] = ()
    _actor_guard: Callable[[], bool] | None = field(default=None, repr=False)
    _replacements: dict[str, tuple[bytes, str]] = field(
        default_factory=dict, init=False, repr=False
    )
    _cleared: set[str] = field(default_factory=set, init=False, repr=False)
    _cancelled: bool = field(default=False, init=False, repr=False)
    _publishing: bool = field(default=False, init=False, repr=False)
    _published: bool = field(default=False, init=False, repr=False)
    _lock: Any = field(default_factory=threading.RLock, init=False, repr=False)

    @property
    def replaced_expression_keys(self) -> tuple[str, ...]:
        """Return replacements in the active version's stable order."""

        with self._lock:
            return tuple(
                str(asset["expression_key"])
                for asset in self.assets
                if asset["expression_key"] in self._replacements
            )

    @property
    def cleared_expression_keys(self) -> tuple[str, ...]:
        """Return clears in the active version's stable order."""

        with self._lock:
            return tuple(
                str(asset["expression_key"])
                for asset in self.assets
                if asset["expression_key"] in self._cleared
            )

    def stage_replacement(
        self, expression_key: str, data: bytes, *, source: str = "manual"
    ) -> None:
        """Stage replacement bytes without touching files or persistence."""

        with self._lock:
            self._ensure_stageable()
            key = self._existing_expression_key(expression_key)
            if not isinstance(data, bytes) or not data:
                raise ValueError("visual_identity_replacement_bytes_invalid")
            if len(data) > MAX_EXPRESSION_ASSET_BYTES:
                raise ValueError("visual_identity_budget_exceeded")
            if source not in {"manual", "upload", "generated"}:
                raise ValueError("visual_identity_replacement_source_invalid")
            projected = sum(
                len(data)
                if str(asset["expression_key"]) == key
                else len(self._replacements[str(asset["expression_key"])][0])
                if str(asset["expression_key"]) in self._replacements
                else int(asset["bytes"])
                for asset in self.assets
                if str(asset["expression_key"]) not in self._cleared
                or str(asset["expression_key"]) == key
            )
            if projected > MAX_EXPRESSION_TOTAL_BYTES:
                raise VisualIdentityPublicationError("visual_identity_budget_exceeded")
            self._replacements[key] = (data, source)
            self._cleared.discard(key)
            self._recompute_default()

    def stage_clear(self, expression_key: str) -> None:
        """Stage one omission without changing the active immutable version."""

        with self._lock:
            self._ensure_stageable()
            key = self._existing_expression_key(expression_key)
            retained = [
                str(asset["expression_key"])
                for asset in self.assets
                if str(asset["expression_key"]) != key
                and str(asset["expression_key"]) not in self._cleared
            ]
            if not retained:
                raise VisualIdentityPublicationError("visual_identity_candidate_empty")
            self._replacements.pop(key, None)
            self._cleared.add(key)
            self._recompute_default()

    def cancel(self) -> None:
        """Make the candidate permanently unpublished."""

        with self._lock:
            if self._published:
                raise VisualIdentityPublicationError(
                    "visual_identity_candidate_published"
                )
            self._cancelled = True

    def _existing_expression_key(self, value: str) -> str:
        key = normalize_expression_key(value)
        if key is None or key not in {
            str(asset["expression_key"]) for asset in self.assets
        }:
            raise ValueError("visual_identity_expression_not_found")
        return key

    def _ensure_editable(self) -> None:
        if self._cancelled:
            raise VisualIdentityPublicationError("visual_identity_candidate_cancelled")
        if self._published:
            raise VisualIdentityPublicationError("visual_identity_candidate_published")

    def _ensure_stageable(self) -> None:
        self._ensure_editable()
        if self._publishing:
            raise VisualIdentityPublicationError("visual_identity_candidate_publishing")

    def _recompute_default(self) -> None:
        retained = [
            str(asset["expression_key"])
            for asset in self.assets
            if str(asset["expression_key"]) not in self._cleared
        ]
        if self.original_default_expression_key in retained:
            self.default_expression_key = self.original_default_expression_key
        elif "neutral" in retained:
            self.default_expression_key = "neutral"
        else:
            self.default_expression_key = retained[0]


@dataclass(frozen=True, slots=True)
class VisualIdentityPublicationResult:
    """Published actor transition consumed by targeted runtime invalidation."""

    actor_kind: str
    actor_id: str
    old_pack_id: int | None
    old_version_id: int | None
    new_pack_id: int
    new_version_id: int
    version_directory: Path


_OPERATIONAL_EXPRESSION_KEYS = {
    "idle": "neutral",
    "thinking": "thinking",
    "speaking": "custom:speaking",
    "error": "custom:error",
}
_LEGACY_OPERATIONAL_STATES = frozenset({"thinking", "speaking", "error"})


def resolve_visual_identity(
    db: Any,
    *,
    actor_kind: str,
    actor_id: int | str,
    requested_state: str,
    manual_expression_key: str | None = None,
    user_data_dir: str | Path | None = None,
) -> VisualIdentityResolution:
    """Resolve one character reaction through pack, legacy, and portrait fallbacks.

    The active graph query is deliberately bounded to the four possible pack
    candidates. Only candidates reached by the fallback walk are read and decoded.
    Returned paths are package/profile-relative identifiers, never filesystem paths.

    Args:
        db: Initialized profile-local ``CharactersRAGDB``.
        actor_kind: Local actor kind. Character portraits are currently supported.
        actor_id: Profile-local actor identifier.
        requested_state: Current Console operational state.
        manual_expression_key: Optional session-local reaction override.
        user_data_dir: Injectable profile root for profile-owned pack assets.

    Returns:
        Frozen resolution metadata and the selected image bytes, if any.
    """
    actor_id_text = str(actor_id)
    requested_state_key = (
        requested_state.strip().lower() if isinstance(requested_state, str) else ""
    )
    requested_key = _OPERATIONAL_EXPRESSION_KEYS.get(requested_state_key)
    if requested_key is None and (
        requested_state_key in CANONICAL_EXPRESSION_SLOTS
        or requested_state_key.startswith(CUSTOM_EXPRESSION_PREFIX)
    ):
        requested_key = normalize_expression_key(requested_state_key)
    manual_key = (
        normalize_expression_key(manual_expression_key)
        if manual_expression_key is not None
        else None
    )
    if actor_kind != "character":
        return _placeholder_resolution(
            actor_kind,
            actor_id_text,
            requested_key,
            manual_key,
            "actor_unavailable",
        )

    legacy_state = (
        requested_state_key
        if requested_state_key in _LEGACY_OPERATIONAL_STATES
        else None
    )
    rows = db.execute_query(
        """
        /* visual_identity_resolver */
        WITH active_graph AS MATERIALIZED (
            SELECT c.id AS actor_row_id,
                   b.id AS binding_id,
                   p.id AS pack_id,
                   p.source_kind AS pack_source_kind,
                   v.id AS pack_version_id,
                   v.default_expression_key AS version_default_expression_key
              FROM character_cards c
              LEFT JOIN visual_identity_bindings b
                ON b.owner_user_id = 0
               AND b.actor_kind = 'character'
               AND b.actor_id = CAST(c.id AS TEXT)
               AND b.status = 'active'
              LEFT JOIN visual_identity_packs p
                ON p.id = b.pack_id
               AND p.owner_user_id = 0
               AND p.status = 'active'
               AND p.active_version_id = b.active_version_id
              LEFT JOIN visual_identity_pack_versions v
                ON v.id = b.active_version_id
               AND v.pack_id = p.id
               AND v.owner_user_id = 0
             WHERE c.id = ? AND c.deleted = 0
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
        SELECT graph.actor_row_id,
               graph.binding_id,
               graph.pack_id,
               graph.pack_source_kind,
               graph.pack_version_id,
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
        (
            actor_id,
            manual_key,
            requested_key,
        ),
    ).fetchall()
    if not rows:
        return _placeholder_resolution(
            actor_kind,
            actor_id_text,
            requested_key,
            manual_key,
            "actor_unavailable",
        )

    candidates = [dict(row) for row in rows]
    for candidate in candidates:
        if candidate["asset_id"] is None:
            continue
        try:
            asset = _resolution_manifest_asset(candidate)
            loaded = load_visual_identity_asset(
                asset,
                source_kind=str(candidate["pack_source_kind"]),
                user_data_dir=user_data_dir,
            )
            _validate_image_bytes(loaded, decoded_pixels_before=0)
        except (TypeError, ValueError, OverflowError) as exc:
            detail = str(exc)
            category = (
                detail
                if detail.startswith("visual_identity_")
                else "visual_identity_asset_metadata_invalid"
            )
            logger.warning(
                "visual_identity_resolution_asset_failed pack_id={} version_id={} "
                "asset_id={} category={}",
                candidate["pack_id"],
                candidate["pack_version_id"],
                candidate["asset_id"],
                category,
            )
            continue
        source, reason = _pack_resolution_category(
            str(candidate["asset_expression_key"]),
            manual_key=manual_key,
            requested_key=requested_key,
            default_key=str(candidate["version_default_expression_key"]),
        )
        digest = str(candidate["asset_sha256"])
        return VisualIdentityResolution(
            actor_kind=actor_kind,
            actor_id=actor_id_text,
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
            cache_identity=_resolution_cache_identity(
                actor_kind,
                actor_id_text,
                requested_key,
                manual_key,
                source,
                f"source_kind={candidate['pack_source_kind']}",
                f"pack_id={candidate['pack_id']}",
                f"pack_version_id={candidate['pack_version_id']}",
                f"asset_id={candidate['asset_id']}",
                f"sha256={digest}",
            ),
            image_bytes=loaded.data,
        )

    if legacy_state is not None:
        legacy = db.execute_query(
            """
            SELECT id, updated_at,
                   CASE WHEN typeof(image) = 'blob'
                              AND length(image) BETWEEN 1 AND ?
                        THEN image END AS image
              FROM character_expression_images
             WHERE character_id = ? AND state_id = ? AND deleted = 0
            """,
            (MAX_EXPRESSION_ASSET_BYTES, actor_id, legacy_state),
        ).fetchone()
    else:
        legacy = None
    if legacy is not None:
        legacy = dict(legacy)
        validated = _validate_fallback_image(legacy["image"])
        if validated is None:
            logger.warning(
                "visual_identity_resolution_fallback_failed actor_kind=character "
                "actor_id={} expression_id={} source=legacy "
                "category=visual_identity_fallback_invalid",
                actor_id_text,
                legacy["id"],
            )
    else:
        validated = None
    if validated is not None:
        data, content_type, is_animated = validated
        digest = hashlib.sha256(data).hexdigest()
        return VisualIdentityResolution(
            actor_kind=actor_kind,
            actor_id=actor_id_text,
            requested_expression_key=requested_key,
            manual_expression_key=manual_key,
            resolved_expression_key=requested_key,
            pack_id=None,
            pack_version_id=None,
            asset_id=None,
            expression_id=int(legacy["id"]),
            storage_source="database",
            storage_relpath=None,
            content_type=content_type,
            is_animated=is_animated,
            resolution_source="legacy_expression",
            fallback_reason="pack_assets_unavailable",
            cache_identity=_resolution_cache_identity(
                actor_kind,
                actor_id_text,
                requested_key,
                manual_key,
                "legacy_expression",
                f"expression_id={legacy['id']}",
                f"updated_at={legacy['updated_at']}",
                f"content_type={content_type}",
                f"is_animated={int(is_animated)}",
                f"sha256={digest}",
            ),
            image_bytes=data,
        )

    card = db.execute_query(
        """SELECT version, length(image) AS image_bytes,
                  CASE WHEN typeof(image) = 'blob'
                             AND length(image) BETWEEN 1 AND ?
                       THEN image END AS image
             FROM character_cards WHERE id = ? AND deleted = 0""",
        (MAX_EXPRESSION_ASSET_BYTES, actor_id),
    ).fetchone()
    if card is not None:
        card = dict(card)
        validated = _validate_fallback_image(card["image"])
        if validated is None and card["image_bytes"] is not None:
            logger.warning(
                "visual_identity_resolution_fallback_failed actor_kind=character "
                "actor_id={} source=card "
                "category=visual_identity_fallback_invalid",
                actor_id_text,
            )
    else:
        validated = None
    if validated is not None:
        data, content_type, is_animated = validated
        digest = hashlib.sha256(data).hexdigest()
        return VisualIdentityResolution(
            actor_kind=actor_kind,
            actor_id=actor_id_text,
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
            resolution_source="card_portrait",
            fallback_reason="legacy_unavailable",
            cache_identity=_resolution_cache_identity(
                actor_kind,
                actor_id_text,
                requested_key,
                manual_key,
                "card_portrait",
                f"card_version={card['version']}",
                f"content_type={content_type}",
                f"is_animated={int(is_animated)}",
                f"sha256={digest}",
            ),
            image_bytes=data,
        )
    return _placeholder_resolution(
        actor_kind,
        actor_id_text,
        requested_key,
        manual_key,
        "portrait_unavailable",
    )


def _resolution_manifest_asset(row: Mapping[str, Any]) -> VisualIdentityManifestAsset:
    return VisualIdentityManifestAsset(
        expression_key=str(row["asset_expression_key"]),
        original_label=str(row["asset_original_expression_key"]),
        display_label=str(row["asset_display_label"]),
        storage_relpath=str(row["asset_storage_relpath"]),
        content_type=str(row["asset_content_type"]),
        bytes=int(row["asset_bytes"]),
        width=int(row["asset_width"]),
        height=int(row["asset_height"]),
        sha256=str(row["asset_sha256"]),
        is_animated=bool(row["asset_is_animated"]),
        frame_count=int(row["asset_frame_count"] or 1),
        duration_ms=(
            int(row["asset_duration_ms"])
            if row["asset_duration_ms"] is not None
            else None
        ),
    )


def _pack_resolution_category(
    expression_key: str,
    *,
    manual_key: str | None,
    requested_key: str | None,
    default_key: str,
) -> tuple[str, str]:
    if manual_key is not None and expression_key == manual_key:
        return "pack_manual", "none"
    if requested_key is not None and expression_key == requested_key:
        reason = "manual_unavailable" if manual_key is not None else "none"
        source = (
            "pack_operational"
            if requested_key in _OPERATIONAL_EXPRESSION_KEYS.values()
            else "pack_explicit"
        )
        return source, reason
    if expression_key == default_key:
        return "pack_default", "requested_unavailable"
    return "pack_neutral", "default_unavailable"


def resolve_historical_visual_identity(
    db: Any,
    *,
    actor_id: int,
    pack_id: int | None,
    pack_version_id: int | None,
    expression_key: str | None,
    expression_id: int | None,
    asset_id: int | None,
    user_data_dir: str | Path | None = None,
) -> VisualIdentityResolution:
    """Resolve only the exact immutable character asset recorded on a message.

    Historical resolution deliberately ignores the actor's current binding. Missing,
    inconsistent, or corrupt references return a deterministic content-free
    placeholder instead of walking current-pack, legacy-image, or portrait fallbacks.
    """

    actor_id_text = str(actor_id)
    normalized_key = normalize_expression_key(expression_key or "")

    def unavailable() -> VisualIdentityResolution:
        return _placeholder_resolution(
            "character",
            actor_id_text,
            normalized_key,
            None,
            "history_unavailable",
        )

    identifiers = (actor_id, pack_id, pack_version_id, asset_id)
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in identifiers
        )
        or (
            expression_id is not None
            and (
                isinstance(expression_id, bool)
                or not isinstance(expression_id, int)
                or expression_id <= 0
            )
        )
        or normalized_key is None
        or expression_key != normalized_key
    ):
        return unavailable()
    try:
        with db.transaction() as connection:
            row = connection.execute(
                """
                SELECT p.source_kind AS pack_source_kind,
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
                  FROM character_cards c
                  JOIN visual_identity_packs p
                    ON p.id = ? AND p.owner_user_id = 0
                  JOIN visual_identity_pack_versions v
                    ON v.id = ? AND v.pack_id = p.id AND v.owner_user_id = 0
                  JOIN visual_identity_assets a
                    ON a.id = ?
                   AND a.pack_id = p.id
                   AND a.pack_version_id = v.id
                   AND a.owner_user_id = 0
                   AND a.deleted = 0
                   AND a.expression_key = ?
                 WHERE c.id = ? AND c.deleted = 0
                """,
                (pack_id, pack_version_id, asset_id, normalized_key, actor_id),
            ).fetchone()
        if row is None:
            return unavailable()
        candidate = dict(row)
        asset = _resolution_manifest_asset(candidate)
        loaded = load_visual_identity_asset(
            asset,
            source_kind=str(candidate["pack_source_kind"]),
            user_data_dir=user_data_dir,
        )
        _validate_image_bytes(loaded, decoded_pixels_before=0)
    except (AttributeError, TypeError, ValueError, OverflowError, sqlite3.Error):
        logger.warning(
            "visual_identity_history_resolution_failed actor_id={} pack_id={} "
            "version_id={} asset_id={} category=history_unavailable",
            actor_id_text,
            pack_id,
            pack_version_id,
            asset_id,
        )
        return unavailable()
    digest = str(candidate["asset_sha256"])
    return VisualIdentityResolution(
        actor_kind="character",
        actor_id=actor_id_text,
        requested_expression_key=normalized_key,
        manual_expression_key=None,
        resolved_expression_key=normalized_key,
        pack_id=pack_id,
        pack_version_id=pack_version_id,
        asset_id=asset_id,
        expression_id=expression_id,
        storage_source=str(candidate["pack_source_kind"]),
        storage_relpath=str(candidate["asset_storage_relpath"]),
        content_type=str(candidate["asset_content_type"]),
        is_animated=bool(candidate["asset_is_animated"]),
        resolution_source="history_immutable",
        fallback_reason="none",
        cache_identity=_resolution_cache_identity(
            "character",
            actor_id_text,
            normalized_key,
            None,
            "history_immutable",
            f"pack_id={pack_id}",
            f"pack_version_id={pack_version_id}",
            f"asset_id={asset_id}",
            f"sha256={digest}",
        ),
        image_bytes=loaded.data,
    )


def _placeholder_resolution(
    actor_kind: str,
    actor_id: str,
    requested_key: str | None,
    manual_key: str | None,
    fallback_reason: str,
) -> VisualIdentityResolution:
    return VisualIdentityResolution(
        actor_kind=actor_kind,
        actor_id=actor_id,
        requested_expression_key=requested_key,
        manual_expression_key=manual_key,
        resolved_expression_key=requested_key,
        pack_id=None,
        pack_version_id=None,
        asset_id=None,
        expression_id=None,
        storage_source="none",
        storage_relpath=None,
        content_type=None,
        is_animated=False,
        resolution_source="placeholder",
        fallback_reason=fallback_reason,
        cache_identity=_resolution_cache_identity(
            actor_kind,
            actor_id,
            requested_key,
            manual_key,
            "placeholder",
            f"fallback_reason={fallback_reason}",
        ),
        image_bytes=None,
    )


def _resolution_cache_identity(
    actor_kind: str,
    actor_id: str,
    requested_key: str | None,
    manual_key: str | None,
    resolution_source: str,
    *parts: object,
) -> tuple[str, ...]:
    return (
        "visual-identity-v1",
        f"actor_kind={actor_kind}",
        f"actor_id={actor_id}",
        f"requested={requested_key or ''}",
        f"manual={manual_key or ''}",
        f"source={resolution_source}",
        *("" if part is None else str(part) for part in parts),
    )


def _validate_fallback_image(value: object) -> tuple[bytes, str, bool] | None:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        return None
    data = bytes(value)
    if not data or len(data) > MAX_EXPRESSION_ASSET_BYTES:
        return None
    try:
        image_format, _, _, is_animated, _, _ = _inspect_image_bytes(data)
        content_type = _IMAGE_CONTENT_TYPES_BY_FORMAT[image_format]
    except (KeyError, ValueError):
        return None
    return data, content_type, is_animated


class _VisualIdentityBudgetError(ValueError):
    """Internal sentinel that preserves the public budget category."""


class _VisualIdentityImageLimitError(ValueError):
    """Internal sentinel that preserves the public decoded-image limit category."""


def compute_pack_content_sha256(
    manifest: Mapping[str, Any] | VisualIdentityManifest,
) -> str:
    """Compute the canonical content digest for a Visual Identity pack.

    Args:
        manifest: Raw or validated manifest data. Only content-bearing fields
            are projected into the canonical payload.

    Returns:
        Lowercase SHA-256 of compact canonical UTF-8 JSON bytes.

    Raises:
        TypeError: If required payload fields cannot be read.
        ValueError: If canonical JSON contains a non-finite numeric value.
    """
    canonical_json = _canonical_pack_content_json(manifest)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _canonical_pack_content_json(
    manifest: Mapping[str, Any] | VisualIdentityManifest,
) -> str:
    """Return the exact canonical JSON hashed for pack content identity."""
    if isinstance(manifest, VisualIdentityManifest):
        schema_id = manifest.schema_id
        pack_id = manifest.pack_id
        default_expression_key = manifest.default_expression_key
        license_id = manifest.license
        assets: Any = manifest.assets
    else:
        schema_id = manifest["schema_id"]
        pack_id = manifest["pack_id"]
        default_expression_key = manifest["default_expression_key"]
        license_id = manifest["license"]
        assets = manifest["assets"]

    projected_assets = sorted(
        (_content_asset_payload(asset) for asset in assets),
        key=lambda asset: asset["original_label"],
    )
    payload = {
        "schema_id": schema_id,
        "pack_id": pack_id,
        "default_expression_key": default_expression_key,
        "license": license_id,
        "assets": projected_assets,
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def validate_visual_identity_manifest(
    data: Mapping[str, Any],
    *,
    require_samira_bundle: bool = False,
    directory_bytes: int | None = None,
) -> VisualIdentityManifest:
    """Validate a Visual Identity pack manifest without reading asset bytes.

    Args:
        data: Parsed manifest mapping.
        require_samira_bundle: Require the exact bundled Samira contract.
        directory_bytes: Measured byte count for the complete supplied pack
            directory. Required when ``require_samira_bundle`` is true or the
            pack ID is the reserved Samira ID; optional for general packs.

    Returns:
        Frozen validated manifest data.

    Raises:
        ValueError: With a stable validation category when data is invalid.
    """
    if not isinstance(data, Mapping):
        raise ValueError("visual_identity_manifest_invalid")

    try:
        schema_id = _nonempty_string(data.get("schema_id"))
        pack_id = _nonempty_string(data.get("pack_id"))
        license_id = _nonempty_string(data.get("license"))
        default_expression_key = _valid_expression_key(
            data.get("default_expression_key")
        )
        title = _nonempty_string(data.get("title", pack_id))
        source_commit_value = data.get("source_server_commit")
        source_server_commit = (
            None
            if source_commit_value is None
            else _nonempty_string(source_commit_value)
        )
        digest = _lower_sha256(data.get("pack_content_sha256"))
        if schema_id != SAMIRA_MANIFEST_SCHEMA_ID or not _LICENSE_RE.fullmatch(
            license_id
        ):
            raise ValueError
        raw_assets = data.get("assets")
        if not isinstance(raw_assets, list) or not raw_assets:
            raise ValueError
        if len(raw_assets) > MAX_EXPRESSION_PACK_ASSETS:
            raise _VisualIdentityBudgetError
        assets = tuple(_validate_manifest_asset(asset) for asset in raw_assets)
        _require_unique_assets(assets)
        if default_expression_key not in {asset.expression_key for asset in assets}:
            raise ValueError
        _validate_directory_bytes(directory_bytes)
        _validate_general_budgets(assets)
    except _VisualIdentityBudgetError:
        raise ValueError("visual_identity_budget_exceeded") from None
    except (KeyError, TypeError, ValueError):
        raise ValueError("visual_identity_manifest_invalid") from None

    manifest = VisualIdentityManifest(
        schema_id=schema_id,
        pack_id=pack_id,
        title=title,
        license=license_id,
        default_expression_key=default_expression_key,
        source_server_commit=source_server_commit,
        pack_content_sha256=digest,
        assets=assets,
    )
    if compute_pack_content_sha256(manifest) != digest:
        raise ValueError("visual_identity_digest_mismatch")

    if require_samira_bundle or pack_id == SAMIRA_PACK_ID:
        if directory_bytes is None:
            raise ValueError("visual_identity_directory_bytes_required")
        _validate_samira_manifest(manifest, directory_bytes=directory_bytes)
    return manifest


def parse_visual_identity_manifest_json(
    raw: bytes | str,
    *,
    require_samira_bundle: bool = False,
    directory_bytes: int | None = None,
) -> VisualIdentityManifest:
    """Parse strict JSON and validate a Visual Identity manifest.

    Args:
        raw: UTF-8 JSON bytes or text.
        require_samira_bundle: Require the exact bundled Samira contract.
        directory_bytes: Measured byte count for the complete supplied pack
            directory. Required when ``require_samira_bundle`` is true or the
            pack ID is the reserved Samira ID; optional for general packs.

    Returns:
        Frozen validated manifest data.

    Raises:
        ValueError: With a stable category for malformed or invalid input.
    """
    try:
        if isinstance(raw, bytes):
            text = raw.decode("utf-8", errors="strict")
        elif isinstance(raw, str):
            text = raw
        else:
            raise TypeError
        data = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
        if not isinstance(data, Mapping):
            raise TypeError
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError):
        raise ValueError("visual_identity_manifest_json_invalid") from None
    return validate_visual_identity_manifest(
        data,
        require_samira_bundle=require_samira_bundle,
        directory_bytes=directory_bytes,
    )


def load_visual_identity_asset(
    asset: VisualIdentityManifestAsset,
    *,
    source_kind: str,
    user_data_dir: str | Path | None = None,
) -> LoadedVisualIdentityAsset:
    """Read and verify one selected immutable Visual Identity asset.

    Args:
        asset: Validated asset metadata.
        source_kind: Owning pack source, either ``builtin`` or profile-owned
            ``manual``.
        user_data_dir: Injectable profile data root for user-owned sources.

    Returns:
        Frozen asset metadata and verified bytes.

    Raises:
        ValueError: With a stable category for unsafe, unavailable, or corrupt
            asset data.
    """
    if not isinstance(asset, VisualIdentityManifestAsset):
        raise ValueError("visual_identity_manifest_invalid")
    if source_kind != "builtin" and source_kind not in _USER_SOURCE_KINDS:
        raise ValueError("visual_identity_source_kind_unsupported")
    if (
        asset.bytes > MAX_EXPRESSION_ASSET_BYTES
        or asset.width > MAX_EXPRESSION_IMAGE_DIMENSION
        or asset.height > MAX_EXPRESSION_IMAGE_DIMENSION
        or asset.frame_count > MAX_EXPRESSION_FRAME_COUNT
        or _decoded_pixels(asset.width, asset.height, asset.frame_count)
        > MAX_EXPRESSION_ASSET_DECODED_PIXELS
    ):
        raise ValueError("visual_identity_budget_exceeded")

    parts = _safe_relative_parts(asset.storage_relpath)
    if source_kind == "builtin":
        data = _read_builtin_asset(parts, expected_bytes=asset.bytes)
    else:
        data = _read_user_asset(
            parts,
            expected_bytes=asset.bytes,
            user_data_dir=user_data_dir,
        )

    if len(data) != asset.bytes:
        raise ValueError("visual_identity_asset_size_mismatch")
    if hashlib.sha256(data).hexdigest() != asset.sha256:
        raise ValueError("visual_identity_asset_sha256_mismatch")
    return LoadedVisualIdentityAsset(asset=asset, data=data)


def _read_builtin_asset(parts: tuple[str, ...], *, expected_bytes: int) -> bytes:
    try:
        candidate = resources.files("tldw_chatbook").joinpath("assets", *parts)
        with candidate.open("rb") as stream:
            return _read_stream_bounded(stream, expected_bytes=expected_bytes)
    except (OSError, RuntimeError, TypeError, AttributeError, zipfile.BadZipFile):
        raise ValueError("visual_identity_asset_unavailable") from None


def _read_user_asset(
    parts: tuple[str, ...],
    *,
    expected_bytes: int,
    user_data_dir: str | Path | None,
) -> bytes:
    assets_root, candidate = _confined_user_asset_path(parts, user_data_dir)
    if _supports_secure_dir_fd():
        return _read_user_asset_secure(
            assets_root,
            parts,
            expected_bytes=expected_bytes,
        )
    return _read_user_asset_fallback(
        assets_root,
        candidate,
        parts,
        expected_bytes=expected_bytes,
    )


def _confined_user_asset_path(
    parts: tuple[str, ...], user_data_dir: str | Path | None
) -> tuple[Path, Path]:
    try:
        profile_root = (
            Path(user_data_dir) if user_data_dir is not None else get_user_data_dir()
        ).resolve(strict=False)
        assets_root = (profile_root / "visual_identities").resolve(strict=False)
        candidate = assets_root.joinpath(*parts)
        resolved_candidate = candidate.resolve(strict=False)
        validate_path(
            candidate,
            assets_root,
            redact_paths=True,
            allow_hidden=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        raise ValueError("visual_identity_path_invalid") from None
    if not assets_root.is_relative_to(
        profile_root
    ) or not resolved_candidate.is_relative_to(assets_root):
        raise ValueError("visual_identity_path_invalid")
    return assets_root, candidate


def _supports_secure_dir_fd() -> bool:
    return (
        os.name == "posix"
        and hasattr(os, "O_NOFOLLOW")
        and hasattr(os, "O_DIRECTORY")
        and os.open in os.supports_dir_fd
        and os.stat in os.supports_dir_fd
        and os.stat in os.supports_follow_symlinks
    )


def _read_user_asset_secure(
    assets_root: Path,
    parts: tuple[str, ...],
    *,
    expected_bytes: int,
) -> bytes:
    opened_fds: list[int] = []
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        root_fd = os.open(assets_root, flags | os.O_DIRECTORY)
        opened_fds.append(root_fd)
        parent_fd = root_fd
        for component in parts[:-1]:
            directory_fd = os.open(
                component,
                flags | os.O_DIRECTORY,
                dir_fd=parent_fd,
            )
            opened_fds.append(directory_fd)
            if not stat.S_ISDIR(os.fstat(directory_fd).st_mode):
                raise ValueError("visual_identity_path_invalid")
            parent_fd = directory_fd

        leaf = parts[-1]
        leaf_fd = os.open(leaf, flags | os.O_NONBLOCK, dir_fd=parent_fd)
        opened_fds.append(leaf_fd)
        opened_stat = os.fstat(leaf_fd)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise ValueError("visual_identity_path_invalid")
        _verify_opened_leaf_identity(parent_fd, leaf, opened_stat)
        data = _read_fd_bounded(leaf_fd, expected_bytes=expected_bytes)
        _verify_opened_leaf_identity(parent_fd, leaf, opened_stat)
        return data
    except ValueError:
        raise
    except OSError as error:
        category = (
            "visual_identity_path_invalid"
            if error.errno in {errno.ELOOP, errno.ENOTDIR}
            else "visual_identity_asset_unavailable"
        )
        raise ValueError(category) from None
    finally:
        for descriptor in reversed(opened_fds):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _verify_opened_leaf_identity(
    parent_fd: int, leaf: str, opened_stat: os.stat_result
) -> None:
    try:
        named_stat = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        raise ValueError("visual_identity_path_invalid") from None
    if (
        not stat.S_ISREG(opened_stat.st_mode)
        or not stat.S_ISREG(named_stat.st_mode)
        or (opened_stat.st_dev, opened_stat.st_ino)
        != (named_stat.st_dev, named_stat.st_ino)
    ):
        raise ValueError("visual_identity_path_invalid")


def _read_user_asset_fallback(
    assets_root: Path,
    candidate: Path,
    parts: tuple[str, ...],
    *,
    expected_bytes: int,
) -> bytes:
    descriptor: int | None = None
    flags = os.O_RDONLY
    if os.name == "posix":
        flags |= getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_BINARY", 0)
    try:
        _verify_fallback_directories(assets_root, parts)
        descriptor = os.open(candidate, flags)
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise ValueError("visual_identity_path_invalid")
        _verify_fallback_directories(assets_root, parts)
        _verify_fallback_identity(candidate, opened_stat)
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            descriptor = None
            _verify_fallback_directories(assets_root, parts)
            _verify_fallback_identity(candidate, opened_stat)
            data = _read_stream_bounded(stream, expected_bytes=expected_bytes)
            _verify_fallback_directories(assets_root, parts)
            _verify_fallback_identity(candidate, opened_stat)
            return data
    except ValueError:
        raise
    except OSError as error:
        category = (
            "visual_identity_path_invalid"
            if error.errno in {errno.ELOOP, errno.ENOTDIR}
            else "visual_identity_asset_unavailable"
        )
        raise ValueError(category) from None
    except (RuntimeError, AttributeError, TypeError):
        raise ValueError("visual_identity_asset_unavailable") from None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _verify_fallback_directories(assets_root: Path, parts: tuple[str, ...]) -> None:
    current = assets_root
    root_stat = os.lstat(current)
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError("visual_identity_path_invalid")
    for component in parts[:-1]:
        current /= component
        if not stat.S_ISDIR(os.lstat(current).st_mode):
            raise ValueError("visual_identity_path_invalid")


def _verify_fallback_identity(candidate: Path, opened_stat: os.stat_result) -> None:
    try:
        named_stat = os.lstat(candidate)
    except OSError:
        raise ValueError("visual_identity_path_invalid") from None
    if (
        not stat.S_ISREG(opened_stat.st_mode)
        or not stat.S_ISREG(named_stat.st_mode)
        or (opened_stat.st_dev, opened_stat.st_ino)
        != (named_stat.st_dev, named_stat.st_ino)
    ):
        raise ValueError("visual_identity_path_invalid")


def _read_fd_bounded(descriptor: int, *, expected_bytes: int) -> bytes:
    return _read_bounded(
        lambda size: os.read(descriptor, size), expected_bytes=expected_bytes
    )


def _read_stream_bounded(stream: Any, *, expected_bytes: int) -> bytes:
    return _read_bounded(stream.read, expected_bytes=expected_bytes)


def _read_bounded(read: Any, *, expected_bytes: int) -> bytes:
    limit = min(expected_bytes, MAX_EXPRESSION_ASSET_BYTES) + 1
    chunks: list[bytes] = []
    byte_count = 0
    while byte_count < limit:
        chunk = read(min(_READ_CHUNK_SIZE, limit - byte_count))
        if not isinstance(chunk, bytes):
            raise ValueError("visual_identity_asset_unavailable")
        if not chunk:
            break
        chunks.append(chunk)
        byte_count += len(chunk)
    return b"".join(chunks)


def validate_visual_identity_assets(
    manifest: VisualIdentityManifest,
    *,
    source_kind: str,
    user_data_dir: str | Path | None = None,
    directory_bytes: int | None = None,
) -> tuple[LoadedVisualIdentityAsset, ...]:
    """Load and completely validate every asset in a candidate pack.

    Args:
        manifest: Previously validated manifest.
        source_kind: Owning pack source kind.
        user_data_dir: Injectable profile data root for user-owned sources.
        directory_bytes: Measured complete supplied-directory byte count.
            Required for the reserved Samira pack ID; optional for general
            packs.

    Returns:
        Every loaded asset in manifest order.

    Raises:
        ValueError: With a stable category for manifest, budget, byte, or image
            validation failures.
    """
    if not isinstance(manifest, VisualIdentityManifest):
        raise ValueError("visual_identity_manifest_invalid")
    try:
        _validate_directory_bytes(directory_bytes)
    except ValueError:
        raise ValueError("visual_identity_manifest_invalid") from None
    if len(manifest.assets) > MAX_EXPRESSION_PACK_ASSETS:
        raise ValueError("visual_identity_budget_exceeded")
    try:
        _validate_general_budgets(manifest.assets)
    except _VisualIdentityBudgetError:
        raise ValueError("visual_identity_budget_exceeded") from None
    if manifest.pack_id == SAMIRA_PACK_ID:
        _validate_samira_manifest(manifest, directory_bytes=directory_bytes)

    loaded_assets: list[LoadedVisualIdentityAsset] = []
    decoded_pixels = 0
    for asset in manifest.assets:
        loaded = load_visual_identity_asset(
            asset,
            source_kind=source_kind,
            user_data_dir=user_data_dir,
        )
        decoded_pixels += _validate_image_bytes(
            loaded,
            decoded_pixels_before=decoded_pixels,
        )
        loaded_assets.append(loaded)
    return tuple(loaded_assets)


def _content_asset_payload(
    asset: Mapping[str, Any] | VisualIdentityManifestAsset,
) -> dict[str, Any]:
    if isinstance(asset, VisualIdentityManifestAsset):
        return {
            "expression_key": asset.expression_key,
            "original_label": asset.original_label,
            "relative_filename": asset.storage_relpath,
            "content_type": asset.content_type,
            "bytes": asset.bytes,
            "width": asset.width,
            "height": asset.height,
            "sha256": asset.sha256,
        }
    relpath = asset.get("storage_relpath", asset.get("relative_filename"))
    return {
        "expression_key": asset["expression_key"],
        "original_label": asset["original_label"],
        "relative_filename": relpath,
        "content_type": asset["content_type"],
        "bytes": asset["bytes"],
        "width": asset["width"],
        "height": asset["height"],
        "sha256": asset["sha256"],
    }


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError


def _nonempty_string(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError:
        raise ValueError from None
    return value


def _positive_int(value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError
    return value


def _lower_sha256(value: Any) -> str:
    if not isinstance(value, str) or _LOWER_SHA256_RE.fullmatch(value) is None:
        raise ValueError
    return value


def _valid_expression_key(value: Any) -> str:
    key = _nonempty_string(value)
    if normalize_expression_key(key) != key:
        raise ValueError
    return key


def _validate_manifest_asset(data: Any) -> VisualIdentityManifestAsset:
    if not isinstance(data, Mapping):
        raise ValueError
    expression_key = _valid_expression_key(data.get("expression_key"))
    original_label = _nonempty_string(data.get("original_label"))
    display_label = _nonempty_string(data.get("display_label"))
    storage_value = data.get("storage_relpath", data.get("relative_filename"))
    if (
        "storage_relpath" in data
        and "relative_filename" in data
        and data["storage_relpath"] != data["relative_filename"]
    ):
        raise ValueError
    storage_relpath = _nonempty_string(storage_value)
    _safe_relative_parts(storage_relpath)
    content_type = _nonempty_string(data.get("content_type"))
    if content_type not in _EXPECTED_IMAGE_FORMATS:
        raise ValueError
    byte_count = _positive_int(data.get("bytes"))
    width = _positive_int(data.get("width"))
    height = _positive_int(data.get("height"))
    sha256 = _lower_sha256(data.get("sha256"))
    is_animated = data.get("is_animated")
    if type(is_animated) is not bool:
        raise ValueError
    frame_count = _positive_int(data.get("frame_count"))
    duration_ms = data.get("duration_ms")
    if is_animated:
        if frame_count <= 1 or type(duration_ms) is not int or duration_ms <= 0:
            raise ValueError
    elif frame_count != 1 or duration_ms is not None:
        raise ValueError

    return VisualIdentityManifestAsset(
        expression_key=expression_key,
        original_label=original_label,
        display_label=display_label,
        storage_relpath=storage_relpath,
        content_type=content_type,
        bytes=byte_count,
        width=width,
        height=height,
        sha256=sha256,
        is_animated=is_animated,
        frame_count=frame_count,
        duration_ms=duration_ms,
    )


def _require_unique_assets(assets: tuple[VisualIdentityManifestAsset, ...]) -> None:
    expression_keys = {asset.expression_key for asset in assets}
    original_labels = {asset.original_label for asset in assets}
    if len(expression_keys) != len(assets) or len(original_labels) != len(assets):
        raise ValueError


def _decoded_pixels(width: int, height: int, frame_count: int) -> int:
    return width * height * frame_count


def _validate_general_budgets(
    assets: tuple[VisualIdentityManifestAsset, ...],
) -> None:
    if any(
        asset.bytes > MAX_EXPRESSION_ASSET_BYTES
        or asset.width > MAX_EXPRESSION_IMAGE_DIMENSION
        or asset.height > MAX_EXPRESSION_IMAGE_DIMENSION
        or asset.frame_count > MAX_EXPRESSION_FRAME_COUNT
        or _decoded_pixels(asset.width, asset.height, asset.frame_count)
        > MAX_EXPRESSION_ASSET_DECODED_PIXELS
        for asset in assets
    ):
        raise _VisualIdentityBudgetError
    if sum(asset.bytes for asset in assets) > MAX_EXPRESSION_TOTAL_BYTES:
        raise _VisualIdentityBudgetError
    if (
        sum(
            _decoded_pixels(asset.width, asset.height, asset.frame_count)
            for asset in assets
        )
        > MAX_EXPRESSION_PACK_DECODED_PIXELS
    ):
        raise _VisualIdentityBudgetError


def _safe_relative_parts(value: str) -> tuple[str, ...]:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\\" in value
        or "\x00" in value
        or PurePosixPath(value).is_absolute()
    ):
        raise ValueError("visual_identity_path_invalid")
    components = value.split("/")
    if any(component in {"", ".", ".."} for component in components):
        raise ValueError("visual_identity_path_invalid")
    return tuple(components)


def _validate_directory_bytes(directory_bytes: int | None) -> None:
    if directory_bytes is not None and (
        type(directory_bytes) is not int or directory_bytes < 0
    ):
        raise ValueError


def _validate_samira_manifest(
    manifest: VisualIdentityManifest,
    *,
    directory_bytes: int | None,
) -> None:
    if directory_bytes is None:
        raise ValueError("visual_identity_directory_bytes_required")
    labels = tuple(asset.original_label for asset in manifest.assets)
    mappings = {asset.original_label: asset.expression_key for asset in manifest.assets}
    exact_contract = (
        manifest.schema_id == SAMIRA_MANIFEST_SCHEMA_ID
        and manifest.pack_id == SAMIRA_PACK_ID
        and manifest.license == SAMIRA_LICENSE
        and manifest.default_expression_key == SAMIRA_DEFAULT_EXPRESSION_KEY
        and manifest.source_server_commit == SAMIRA_SERVER_COMMIT
        and labels == SAMIRA_REACTION_LABELS
        and mappings == SAMIRA_EXPRESSION_KEYS
        and all(
            asset.storage_relpath
            == f"characters/samira/expressions/{asset.original_label}.webp"
            and asset.content_type == "image/webp"
            and asset.width == 1024
            and asset.height == 1024
            and not asset.is_animated
            and asset.frame_count == 1
            and asset.duration_ms is None
            for asset in manifest.assets
        )
    )
    if not exact_contract:
        raise ValueError("visual_identity_samira_contract_invalid")
    if any(asset.bytes > SAMIRA_MAX_REACTION_BYTES for asset in manifest.assets):
        raise ValueError("visual_identity_budget_exceeded")
    if sum(asset.bytes for asset in manifest.assets) > SAMIRA_MAX_REACTIONS_BYTES:
        raise ValueError("visual_identity_budget_exceeded")
    if directory_bytes > SAMIRA_MAX_DIRECTORY_BYTES:
        raise ValueError("visual_identity_budget_exceeded")


def _validate_image_bytes(
    loaded: LoadedVisualIdentityAsset,
    *,
    decoded_pixels_before: int = 0,
) -> int:
    asset = loaded.asset
    (
        image_format,
        image_size,
        frame_count,
        is_animated,
        duration_ms,
        decoded_pixels,
    ) = _inspect_image_bytes(
        loaded.data,
        decoded_pixels_before=decoded_pixels_before,
    )

    try:
        expected_format = _EXPECTED_IMAGE_FORMATS[asset.content_type]
    except KeyError:
        raise ValueError("visual_identity_asset_format_mismatch") from None
    if image_format != expected_format:
        raise ValueError("visual_identity_asset_format_mismatch")
    if image_size != (asset.width, asset.height):
        raise ValueError("visual_identity_asset_dimensions_mismatch")
    if (
        frame_count != asset.frame_count
        or is_animated != asset.is_animated
        or duration_ms != asset.duration_ms
    ):
        raise ValueError("visual_identity_asset_frame_mismatch")
    return decoded_pixels


def _inspect_image_bytes(
    data: bytes,
    *,
    decoded_pixels_before: int = 0,
) -> tuple[str, tuple[int, int], int, bool, int | None, int]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                image_format = image.format or ""
                image_size = image.size
                frame_count = max(int(getattr(image, "n_frames", 1) or 1), 1)
                is_animated = (
                    bool(getattr(image, "is_animated", False)) or frame_count > 1
                )
                if (
                    image_size[0] > MAX_EXPRESSION_IMAGE_DIMENSION
                    or image_size[1] > MAX_EXPRESSION_IMAGE_DIMENSION
                    or frame_count > MAX_EXPRESSION_FRAME_COUNT
                ):
                    raise _VisualIdentityImageLimitError
                decoded_pixels = _decoded_pixels(
                    image_size[0], image_size[1], frame_count
                )
                if (
                    decoded_pixels > MAX_EXPRESSION_ASSET_DECODED_PIXELS
                    or decoded_pixels_before + decoded_pixels
                    > MAX_EXPRESSION_PACK_DECODED_PIXELS
                ):
                    raise _VisualIdentityBudgetError
                decoded_duration_ms = _image_duration_ms(image, frame_count)
                duration_ms = decoded_duration_ms if is_animated else None
    except _VisualIdentityImageLimitError:
        raise ValueError("visual_identity_asset_limits_exceeded") from None
    except _VisualIdentityBudgetError:
        raise ValueError("visual_identity_budget_exceeded") from None
    except (
        OSError,
        EOFError,
        SyntaxError,
        RuntimeError,
        IndexError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        ValueError,
    ):
        raise ValueError("visual_identity_asset_decode_invalid") from None
    return (
        image_format,
        image_size,
        frame_count,
        is_animated,
        duration_ms,
        decoded_pixels,
    )


def _image_duration_ms(image: Image.Image, frame_count: int) -> int:
    duration_ms = 0
    for frame_index in range(frame_count):
        image.seek(frame_index)
        image.load()
        duration_ms += int(image.info.get("duration") or 0)
    return duration_ms


def create_visual_identity_candidate(
    db: Any,
    *,
    actor_kind: str,
    actor_id: int | str,
    actor_authority: tuple[str, ...] = (),
    actor_guard: Callable[[], bool] | None = None,
) -> VisualIdentityCandidate:
    """Snapshot one active immutable graph for in-memory copy-on-write edits."""

    from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository

    if actor_kind not in {"character", "persona"}:
        raise ValueError("visual_identity_actor_kind_invalid")
    if actor_kind == "persona":
        if (
            type(actor_authority) is not tuple
            or not actor_authority
            or any(type(item) is not str or not item for item in actor_authority)
            or not callable(actor_guard)
        ):
            raise ValueError("visual_identity_actor_changed")
        try:
            actor_current = actor_guard()
        except Exception:
            actor_current = False
        if actor_current is not True:
            raise ValueError("visual_identity_actor_changed")
    graph = VisualIdentityRepository(db).get_active_actor_pack(actor_kind, actor_id)
    if graph is None and actor_kind == "persona":
        return VisualIdentityCandidate(
            actor_kind="persona",
            actor_id=str(actor_id),
            old_pack_id=None,
            old_version_id=None,
            old_binding_id=None,
            old_binding_version=None,
            old_pack_version=None,
            source_kind="manual",
            title="Persona reactions",
            description="",
            default_expression_key="neutral",
            original_default_expression_key="neutral",
            source_context={"source_id": "persona.local"},
            assets=_empty_persona_candidate_assets(),
            actor_authority=actor_authority,
            _actor_guard=actor_guard,
        )
    if graph is None or not graph["assets"]:
        raise ValueError("visual_identity_active_pack_not_found")
    pack = graph["pack"]
    if pack["source_kind"] not in {"builtin", "manual"}:
        raise ValueError("visual_identity_source_kind_unsupported")
    try:
        source_context = json.loads(
            pack["source_context_json"], parse_constant=_reject_json_constant
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        raise ValueError("visual_identity_source_context_invalid") from None
    if not isinstance(source_context, dict):
        raise ValueError("visual_identity_source_context_invalid")
    return VisualIdentityCandidate(
        actor_kind=actor_kind,
        actor_id=str(actor_id),
        old_pack_id=int(pack["id"]),
        old_version_id=int(graph["version"]["id"]),
        old_binding_id=int(graph["binding"]["id"]),
        old_binding_version=int(graph["binding"]["version"]),
        old_pack_version=int(pack["version"]),
        source_kind=str(pack["source_kind"]),
        title=str(pack["title"]),
        description=str(pack["description"]),
        default_expression_key=str(graph["version"]["default_expression_key"]),
        original_default_expression_key=str(graph["version"]["default_expression_key"]),
        source_context=dict(source_context),
        assets=tuple(dict(asset) for asset in graph["assets"]),
        actor_authority=actor_authority,
        _actor_guard=actor_guard,
    )


def _empty_persona_candidate_assets() -> tuple[dict[str, Any], ...]:
    """Return canonical metadata slots for an unpublished Persona pack."""

    return tuple(
        {
            "id": None,
            "expression_key": key,
            "original_expression_key": key,
            "display_label": display_label_for_expression_key(key),
            "source_filename": "",
            "storage_relpath": "",
            "content_type": "",
            "bytes": 0,
            "sha256": "",
            "width": 0,
            "height": 0,
            "source_context_json": "{}",
            "is_animated": False,
            "frame_count": 1,
            "duration_ms": None,
        }
        for key in CANONICAL_EXPRESSION_SLOTS
    )


def publish_visual_identity_candidate(
    db: Any,
    candidate: VisualIdentityCandidate,
    *,
    user_data_dir: str | Path | None = None,
    atomic_replace: Callable[..., None] = os.replace,
) -> VisualIdentityPublicationResult:
    """Publish one complete candidate directory and one immutable DB version."""

    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDBError
    from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository

    if not isinstance(candidate, VisualIdentityCandidate):
        raise VisualIdentityPublicationError("visual_identity_candidate_invalid")
    try:
        if db.get_connection().in_transaction:
            raise VisualIdentityPublicationError("visual_identity_transaction_active")
    except VisualIdentityPublicationError:
        raise
    except (AttributeError, CharactersRAGDBError, sqlite3.Error, RuntimeError):
        raise VisualIdentityPublicationError(
            "visual_identity_database_failed"
        ) from None
    with candidate._lock:
        candidate._ensure_stageable()
        if not candidate._replacements and not candidate._cleared:
            raise VisualIdentityPublicationError("visual_identity_candidate_clean")
        if not callable(atomic_replace):
            raise VisualIdentityPublicationError("visual_identity_candidate_invalid")
        if candidate._actor_guard is not None:
            try:
                actor_current = candidate._actor_guard()
            except Exception:
                actor_current = False
            if actor_current is not True:
                raise VisualIdentityPublicationError("visual_identity_actor_changed")
        candidate._publishing = True

    repository = VisualIdentityRepository(db)
    unbound = candidate.old_pack_id is None
    try:
        live = repository.get_active_actor_pack(
            candidate.actor_kind, candidate.actor_id
        )
        active_binding_count = (
            0
            if unbound
            else repository.count_active_pack_bindings(candidate.old_pack_id)
        )
    except (
        CharactersRAGDBError,
        sqlite3.Error,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        _reset_candidate_publication(candidate)
        raise VisualIdentityPublicationError(
            "visual_identity_database_failed"
        ) from None
    binding_changed = live is not None if unbound else live is None
    if not unbound and live is not None:
        binding_changed = (
            int(live["binding"]["id"]),
            int(live["pack"]["id"]),
            int(live["version"]["id"]),
            int(live["binding"]["version"]),
            int(live["pack"]["version"]),
        ) != (
            candidate.old_binding_id,
            candidate.old_pack_id,
            candidate.old_version_id,
            candidate.old_binding_version,
            candidate.old_pack_version,
        )
    if binding_changed:
        _reset_candidate_publication(candidate)
        raise VisualIdentityPublicationError("visual_identity_binding_changed")

    try:
        profile_root, assets_root = _visual_identity_publication_roots(user_data_dir)
        fork_pack = (
            unbound or candidate.source_kind == "builtin" or active_binding_count > 1
        )
        profile_pack_token = _publication_pack_token(candidate, force_new=fork_pack)
    except VisualIdentityPublicationError:
        _reset_candidate_publication(candidate)
        raise
    publication_token = uuid4().hex
    versions_root = assets_root / "packs" / profile_pack_token / "versions"
    staging_name = f".staging-{publication_token}"
    final_name = publication_token
    staging_dir = versions_root / staging_name
    final_dir = versions_root / final_name
    staging_relpath = staging_dir.relative_to(assets_root).as_posix()
    final_relpath = final_dir.relative_to(assets_root).as_posix()
    posix_guards = _publication_posix_guards_available()
    chain: list[tuple[int, str, int, Path]] = []
    secured_identities: dict[Path, tuple[int, int]] = {}
    staging_identity: tuple[int, int] | None = None
    versions_fd = -1
    staging_fd = -1

    def retained_candidate_relpath() -> str | None:
        return _remaining_publication_candidate_relpath(
            posix_guards=posix_guards,
            versions_fd=versions_fd,
            staging_fd=staging_fd,
            final_name=final_name,
            final_relpath=final_relpath,
        )

    def discard_unpublished_staging(retained_relpath: str | None) -> str | None:
        if retained_relpath is None and posix_guards and versions_fd >= 0:
            if not _discard_staging_directory(versions_fd, staging_name, staging_fd):
                if _entry_matches_fd(versions_fd, staging_name, staging_fd):
                    return staging_relpath
        return retained_relpath

    try:
        for directory in (
            profile_root,
            assets_root,
            assets_root / "packs",
            versions_root.parent,
            versions_root,
        ):
            privacy = secure_private_directory(
                directory, create=True, application_owned=True
            )
            if not (privacy.verified_private if posix_guards else privacy.usable):
                raise PermissionError
            secured_identities[directory] = _path_identity(directory)
        if posix_guards:
            try:
                chain = _open_publication_chain(versions_root)
            except OSError:
                raise PermissionError from None
            versions_fd = chain[-1][2]
            if not _publication_chain_matches(chain, secured_identities):
                raise PermissionError
        staging_privacy = secure_private_directory(
            staging_dir, create=True, application_owned=True
        )
        if not (
            staging_privacy.verified_private if posix_guards else staging_privacy.usable
        ):
            raise PermissionError
        staging_identity = _path_identity(staging_dir)
        if posix_guards:
            staging_fd = os.open(
                staging_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=versions_fd,
            )

            def write_file(name, data):
                return _write_private_publication_file(staging_fd, name, data)

            def read_file(name, limit):
                return _read_private_publication_file(staging_fd, name, max_bytes=limit)

        else:

            def write_file(name, data):
                return _write_private_publication_path(staging_dir, name, data)

            def read_file(name, limit):
                return _read_private_publication_path(staging_dir, name, limit)

        assets, manifest = _materialize_visual_identity_candidate(
            candidate,
            write_file=write_file,
            final_relpath=final_relpath,
            profile_pack_token=profile_pack_token,
            user_data_dir=profile_root,
        )
        manifest_raw = json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        write_file("manifest.json", manifest_raw)
        _verify_materialized_candidate(read_file, assets)
        _sync_publication_directory(staging_fd if posix_guards else staging_dir)
        with candidate._lock:
            candidate._ensure_editable()
            namespace_current = (
                _publication_chain_matches(chain, secured_identities)
                if posix_guards
                else _path_chain_matches(secured_identities, profile_root=profile_root)
            )
            if not namespace_current:
                raise PermissionError
            if posix_guards:
                atomic_replace(
                    staging_name,
                    final_name,
                    src_dir_fd=versions_fd,
                    dst_dir_fd=versions_fd,
                )
            else:
                atomic_replace(staging_dir, final_dir)
            namespace_current = (
                _publication_chain_matches(chain, secured_identities)
                and _entry_matches_fd(versions_fd, final_name, staging_fd)
                if posix_guards
                else _path_chain_matches(secured_identities, profile_root=profile_root)
                and staging_identity is not None
                and _path_matches_identity(final_dir, staging_identity)
            )
            if not namespace_current:
                if posix_guards:
                    _discard_pinned_directory(versions_fd, final_name, staging_fd)
                raise VisualIdentityPublicationError(
                    "visual_identity_publication_denied"
                )
            published_read = (
                read_file
                if posix_guards
                else lambda name, limit: _read_private_publication_path(
                    final_dir, name, limit
                )
            )
            _verify_materialized_candidate(published_read, assets)
            _sync_publication_directory(versions_fd if posix_guards else versions_root)

            actor_denied = False

            def publication_guard() -> bool:
                nonlocal actor_denied
                if posix_guards:
                    filesystem_current = _publication_chain_matches(
                        chain, secured_identities
                    ) and _entry_matches_fd(versions_fd, final_name, staging_fd)
                else:
                    filesystem_current = (
                        _path_chain_matches(
                            secured_identities, profile_root=profile_root
                        )
                        and staging_identity is not None
                        and _path_matches_identity(final_dir, staging_identity)
                    )
                if not filesystem_current:
                    return False
                if candidate._actor_guard is None:
                    return True
                try:
                    actor_current = candidate._actor_guard()
                except Exception:
                    actor_current = False
                if actor_current is not True:
                    actor_denied = True
                    return False
                return True

            try:
                if fork_pack:
                    source_context = (
                        {
                            "profile_pack_id": profile_pack_token,
                            "source_id": "persona.local",
                        }
                        if unbound
                        else {
                            "profile_pack_id": profile_pack_token,
                            "forked_from_pack_id": candidate.old_pack_id,
                            "forked_from_version_id": candidate.old_version_id,
                        }
                    )
                    graph = repository.activate_pack(
                        pack={
                            "title": (
                                candidate.title
                                if unbound
                                else f"{candidate.title} (Profile Copy)"
                            ),
                            "description": candidate.description,
                            "default_expression_key": candidate.default_expression_key,
                            "source_kind": "manual",
                            "source_context": source_context,
                        },
                        manifest=manifest,
                        assets=assets,
                        actor_kind=candidate.actor_kind,
                        actor_id=candidate.actor_id,
                        expected_active_identity=(
                            None
                            if unbound
                            else (candidate.old_pack_id, candidate.old_version_id)
                        ),
                        expected_binding_id=candidate.old_binding_id,
                        expected_binding_version=candidate.old_binding_version,
                        expected_source_pack_version=candidate.old_pack_version,
                        require_unbound_actor=unbound,
                        publication_guard=publication_guard,
                    )
                else:
                    graph = repository.publish_version(
                        candidate.old_pack_id,
                        manifest=manifest,
                        assets=assets,
                        actor_kind=candidate.actor_kind,
                        actor_id=candidate.actor_id,
                        default_expression_key=candidate.default_expression_key,
                        expected_active_version_id=candidate.old_version_id,
                        expected_binding_id=candidate.old_binding_id,
                        expected_binding_version=candidate.old_binding_version,
                        expected_pack_version=candidate.old_pack_version,
                        require_single_active_binding=True,
                        publication_guard=publication_guard,
                    )
            except ValueError as error:
                if str(error) == "visual_identity_binding_changed":
                    category = "visual_identity_binding_changed"
                elif str(error) == "visual_identity_publication_changed":
                    category = (
                        "visual_identity_actor_changed"
                        if actor_denied
                        else "visual_identity_publication_denied"
                    )
                else:
                    category = "visual_identity_database_failed"
                raise VisualIdentityPublicationError(category) from None
            except (
                CharactersRAGDBError,
                sqlite3.Error,
                OSError,
                RuntimeError,
                TypeError,
            ):
                raise VisualIdentityPublicationError(
                    "visual_identity_database_failed"
                ) from None
            candidate._published = True
            candidate._publishing = False
    except VisualIdentityPublicationError as error:
        retained_relpath = retained_candidate_relpath()
        retained_relpath = discard_unpublished_staging(retained_relpath)
        _reset_candidate_publication(candidate)
        if error.cleanup_candidate_relpath is None and retained_relpath is not None:
            raise VisualIdentityPublicationError(
                error.category, cleanup_candidate_relpath=retained_relpath
            ) from None
        raise
    except PermissionError:
        retained_relpath = retained_candidate_relpath()
        retained_relpath = discard_unpublished_staging(retained_relpath)
        _reset_candidate_publication(candidate)
        raise VisualIdentityPublicationError(
            "visual_identity_publication_denied",
            cleanup_candidate_relpath=retained_relpath,
        ) from None
    except (OSError, TypeError, ValueError, OverflowError) as error:
        retained_relpath = retained_candidate_relpath()
        retained_relpath = discard_unpublished_staging(retained_relpath)
        _reset_candidate_publication(candidate)
        raise VisualIdentityPublicationError(
            (
                "visual_identity_publication_failed"
                if isinstance(error, OSError) and retained_relpath == final_relpath
                else "visual_identity_candidate_invalid"
            ),
            cleanup_candidate_relpath=retained_relpath,
        ) from None
    finally:
        if staging_fd >= 0:
            os.close(staging_fd)
        if chain:
            _close_publication_chain(chain)
    return VisualIdentityPublicationResult(
        actor_kind=candidate.actor_kind,
        actor_id=candidate.actor_id,
        old_pack_id=candidate.old_pack_id,
        old_version_id=candidate.old_version_id,
        new_pack_id=int(graph["pack"]["id"]),
        new_version_id=int(graph["version"]["id"]),
        version_directory=final_dir,
    )


_PUBLICATION_CLEANUP_RE = re.compile(
    r"\Apacks/(profile-[0-9a-f]{32})/versions/"
    r"([0-9a-f]{32}|\.staging-[0-9a-f]{32})\Z"
)


def cleanup_visual_identity_publication_candidate(
    db: Any,
    cleanup_candidate_relpath: str,
    *,
    user_data_dir: str | Path | None = None,
) -> bool:
    """Delete one unreferenced POSIX publication directory without following links."""

    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDBError

    if not isinstance(cleanup_candidate_relpath, str):
        raise VisualIdentityPublicationError("visual_identity_cleanup_denied")
    match = _PUBLICATION_CLEANUP_RE.fullmatch(cleanup_candidate_relpath)
    if match is None or not _publication_posix_guards_available():
        raise VisualIdentityPublicationError("visual_identity_cleanup_denied")
    profile_root, assets_root = _visual_identity_publication_roots(user_data_dir)
    connection = None
    try:
        connection = db.get_connection()
        if connection.in_transaction:
            raise VisualIdentityPublicationError("visual_identity_transaction_active")
    except VisualIdentityPublicationError:
        raise
    except (
        AttributeError,
        CharactersRAGDBError,
        sqlite3.Error,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        raise VisualIdentityPublicationError(
            "visual_identity_database_failed"
        ) from None
    pack_token, version_token = match.groups()
    versions_root = assets_root / "packs" / pack_token / "versions"
    identities: dict[Path, tuple[int, int]] = {}
    try:
        for directory in (
            profile_root,
            assets_root,
            assets_root / "packs",
            versions_root.parent,
            versions_root,
        ):
            privacy = secure_private_directory(
                directory, create=False, application_owned=True
            )
            if not privacy.verified_private:
                raise VisualIdentityPublicationError("visual_identity_cleanup_denied")
            identities[directory] = _path_identity(directory)
    except VisualIdentityPublicationError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError):
        raise VisualIdentityPublicationError("visual_identity_cleanup_denied") from None
    chain: list[tuple[int, str, int, Path]] = []
    candidate_fd = -1
    reservation_active = False
    try:
        chain = _open_publication_chain(versions_root)
        versions_fd = chain[-1][2]
        if not _publication_chain_matches(chain, identities):
            raise VisualIdentityPublicationError("visual_identity_cleanup_denied")
        candidate_fd = os.open(
            version_token,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=versions_fd,
        )
        connection.execute("BEGIN IMMEDIATE")
        reservation_active = True
        referenced = connection.execute(
            """
            SELECT 1
             FROM visual_identity_assets
             WHERE owner_user_id = 0
               AND (
                    storage_relpath = ? OR storage_relpath LIKE ?
                    OR preview_relpath = ? OR preview_relpath LIKE ?
               )
             LIMIT 1
            """,
            (
                cleanup_candidate_relpath,
                f"{cleanup_candidate_relpath}/%",
                cleanup_candidate_relpath,
                f"{cleanup_candidate_relpath}/%",
            ),
        ).fetchone()
        if referenced is not None:
            raise VisualIdentityPublicationError("visual_identity_cleanup_referenced")
        if not _publication_chain_matches(chain, identities):
            raise VisualIdentityPublicationError("visual_identity_cleanup_denied")
        if not _discard_pinned_directory(versions_fd, version_token, candidate_fd):
            raise VisualIdentityPublicationError("visual_identity_cleanup_denied")
        _sync_publication_directory(versions_fd)
        referenced = connection.execute(
            """
            SELECT 1
             FROM visual_identity_assets
             WHERE owner_user_id = 0
               AND (
                    storage_relpath = ? OR storage_relpath LIKE ?
                    OR preview_relpath = ? OR preview_relpath LIKE ?
               )
             LIMIT 1
            """,
            (
                cleanup_candidate_relpath,
                f"{cleanup_candidate_relpath}/%",
                cleanup_candidate_relpath,
                f"{cleanup_candidate_relpath}/%",
            ),
        ).fetchone()
        if referenced is not None:
            raise VisualIdentityPublicationError("visual_identity_cleanup_referenced")
        connection.commit()
        reservation_active = False
        return True
    except VisualIdentityPublicationError:
        raise
    except sqlite3.Error:
        raise VisualIdentityPublicationError(
            "visual_identity_database_failed"
        ) from None
    except OSError:
        raise VisualIdentityPublicationError("visual_identity_cleanup_denied") from None
    finally:
        if reservation_active:
            connection.rollback()
        if candidate_fd >= 0:
            os.close(candidate_fd)
        if chain:
            _close_publication_chain(chain)


def _visual_identity_publication_roots(
    user_data_dir: str | Path | None,
) -> tuple[Path, Path]:
    try:
        profile_root = (
            Path(user_data_dir) if user_data_dir is not None else get_user_data_dir()
        ).resolve(strict=False)
        package_resource = resources.files("tldw_chatbook")
        package_root = (
            Path(package_resource).resolve(strict=False)
            if isinstance(package_resource, os.PathLike)
            else None
        )
    except (OSError, RuntimeError, TypeError):
        raise VisualIdentityPublicationError("visual_identity_path_invalid") from None
    assets_root = (profile_root / "visual_identities").resolve(strict=False)
    if not assets_root.is_relative_to(profile_root):
        raise VisualIdentityPublicationError("visual_identity_path_invalid")
    if package_root is not None and (
        assets_root.is_relative_to(package_root)
        or package_root.is_relative_to(assets_root)
    ):
        raise VisualIdentityPublicationError("visual_identity_package_root_immutable")
    return profile_root, assets_root


def _publication_pack_token(
    candidate: VisualIdentityCandidate, *, force_new: bool = False
) -> str:
    if force_new:
        return f"profile-{uuid4().hex}"
    value = candidate.source_context.get("profile_pack_id")
    if (
        not isinstance(value, str)
        or not value.startswith("profile-")
        or _safe_relative_parts(value) != (value,)
    ):
        raise VisualIdentityPublicationError("visual_identity_source_context_invalid")
    return value


def _materialize_visual_identity_candidate(
    candidate: VisualIdentityCandidate,
    *,
    write_file: Callable[[str, bytes], None],
    final_relpath: str,
    profile_pack_token: str,
    user_data_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    decoded_pixels = 0
    materialized_bytes = 0
    for index, stored in enumerate(candidate.assets):
        expression_key = str(stored["expression_key"])
        if expression_key in candidate._cleared:
            continue
        replacement = candidate._replacements.get(expression_key)
        if replacement is None and stored.get("id") is None:
            continue
        if replacement is None:
            source_asset = _manifest_asset_from_row(stored)
            loaded = load_visual_identity_asset(
                source_asset,
                source_kind=candidate.source_kind,
                user_data_dir=user_data_dir,
            )
            decoded_pixels += _validate_image_bytes(
                loaded, decoded_pixels_before=decoded_pixels
            )
            data = loaded.data
            content_type = source_asset.content_type
            width, height = source_asset.width, source_asset.height
            is_animated = source_asset.is_animated
            frame_count = source_asset.frame_count
            duration_ms = source_asset.duration_ms
            source_context = {"retained_asset_id": int(stored["id"])}
        else:
            data, replacement_source = replacement
            (
                image_format,
                (width, height),
                frame_count,
                is_animated,
                duration_ms,
                pixels,
            ) = _inspect_image_bytes(data, decoded_pixels_before=decoded_pixels)
            decoded_pixels += pixels
            try:
                content_type = _IMAGE_CONTENT_TYPES_BY_FORMAT[image_format]
            except KeyError:
                raise ValueError("visual_identity_asset_format_mismatch") from None
            source_context = {"publication_source": replacement_source}
        materialized_bytes += len(data)
        if materialized_bytes > MAX_EXPRESSION_TOTAL_BYTES:
            raise VisualIdentityPublicationError("visual_identity_budget_exceeded")
        extension = {
            "image/gif": "gif",
            "image/jpeg": "jpg",
            "image/png": "png",
            "image/webp": "webp",
        }[content_type]
        filename = (
            f"{index:03d}-{_sanitize_expression_token(str(stored['original_expression_key']))}"
            f".{extension}"
        )
        data_digest = hashlib.sha256(data).hexdigest()
        storage_relpath = f"{final_relpath}/{filename}"
        asset = {
            "expression_key": expression_key,
            "original_expression_key": str(stored["original_expression_key"]),
            "display_label": str(stored["display_label"]),
            "source_filename": filename,
            "storage_relpath": storage_relpath,
            "content_type": content_type,
            "bytes": len(data),
            "sha256": data_digest,
            "width": width,
            "height": height,
            "source_context": source_context,
            "is_animated": is_animated,
            "frame_count": frame_count,
            "duration_ms": duration_ms,
        }
        write_file(filename, data)
        prepared.append(asset)

    manifest = {
        "schema_id": SAMIRA_MANIFEST_SCHEMA_ID,
        "pack_id": f"tldw.profile.{profile_pack_token}",
        "title": candidate.title,
        "license": SAMIRA_LICENSE,
        "default_expression_key": candidate.default_expression_key,
        "source_server_commit": None,
        "pack_content_sha256": "0" * 64,
        "assets": [
            {
                "expression_key": asset["expression_key"],
                "original_label": asset["original_expression_key"],
                "display_label": asset["display_label"],
                "storage_relpath": asset["storage_relpath"],
                "content_type": asset["content_type"],
                "bytes": asset["bytes"],
                "width": asset["width"],
                "height": asset["height"],
                "sha256": asset["sha256"],
                "is_animated": asset["is_animated"],
                "frame_count": asset["frame_count"],
                "duration_ms": asset["duration_ms"],
            }
            for asset in prepared
        ],
    }
    manifest["pack_content_sha256"] = compute_pack_content_sha256(manifest)
    validate_visual_identity_manifest(manifest)
    return prepared, manifest


def _manifest_asset_from_row(row: Mapping[str, Any]) -> VisualIdentityManifestAsset:
    return VisualIdentityManifestAsset(
        expression_key=str(row["expression_key"]),
        original_label=str(row["original_expression_key"]),
        display_label=str(row["display_label"]),
        storage_relpath=str(row["storage_relpath"]),
        content_type=str(row["content_type"]),
        bytes=int(row["bytes"]),
        width=int(row["width"]),
        height=int(row["height"]),
        sha256=str(row["sha256"]),
        is_animated=bool(row["is_animated"]),
        frame_count=int(row["frame_count"] or 1),
        duration_ms=(int(row["duration_ms"]) if row["duration_ms"] else None),
    )


def _write_private_publication_file(
    directory_fd: int, filename: str, data: bytes
) -> None:
    if _safe_relative_parts(filename) != (filename,):
        raise ValueError("visual_identity_candidate_invalid")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(filename, flags, 0o600, dir_fd=directory_fd)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _verify_materialized_candidate(
    read_file: Callable[[str, int], bytes], assets: list[dict[str, Any]]
) -> None:
    decoded_pixels = 0
    for asset in assets:
        data = read_file(str(asset["source_filename"]), MAX_EXPRESSION_ASSET_BYTES)
        loaded = LoadedVisualIdentityAsset(
            asset=_manifest_asset_from_row(asset), data=data
        )
        if len(data) != loaded.asset.bytes or hashlib.sha256(data).hexdigest() != (
            loaded.asset.sha256
        ):
            raise ValueError("visual_identity_candidate_invalid")
        decoded_pixels += _validate_image_bytes(
            loaded, decoded_pixels_before=decoded_pixels
        )


def _publication_posix_guards_available() -> bool:
    return (
        os.name == "posix"
        and getattr(os, "O_DIRECTORY", 0) != 0
        and getattr(os, "O_NOFOLLOW", 0) != 0
        # CPython exposes os.replace through the same descriptor-relative
        # implementation as os.rename, but only registers os.rename here.
        and {os.open, os.stat, os.rename, os.unlink}.issubset(os.supports_dir_fd)
        and os.stat in os.supports_follow_symlinks
    )


def _open_publication_chain(
    path: Path,
) -> list[tuple[int, str, int, Path]]:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    parent_fd = os.open(os.sep, flags)
    chain: list[tuple[int, str, int, Path]] = []
    current_path = Path(os.sep)
    try:
        for component in path.parts[1:]:
            child_fd = os.open(component, flags, dir_fd=parent_fd)
            current_path /= component
            chain.append((parent_fd, component, child_fd, current_path))
            parent_fd = child_fd
    except BaseException:
        _close_publication_chain(chain, root_fd=parent_fd if not chain else None)
        raise
    return chain


def _close_publication_chain(
    chain: list[tuple[int, str, int, Path]], *, root_fd: int | None = None
) -> None:
    descriptors = {root_fd} if root_fd is not None else set()
    for parent_fd, _name, child_fd, _path in chain:
        descriptors.add(parent_fd)
        descriptors.add(child_fd)
    for descriptor in sorted(
        (fd for fd in descriptors if fd is not None), reverse=True
    ):
        try:
            os.close(descriptor)
        except OSError:
            pass


def _publication_chain_matches(
    chain: list[tuple[int, str, int, Path]],
    secured_identities: Mapping[Path, tuple[int, int]],
) -> bool:
    by_path = {path: child_fd for _parent, _name, child_fd, path in chain}
    try:
        for parent_fd, name, child_fd, _path in chain:
            entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            opened = os.fstat(child_fd)
            if not stat.S_ISDIR(entry.st_mode) or (entry.st_dev, entry.st_ino) != (
                opened.st_dev,
                opened.st_ino,
            ):
                return False
        for path, identity in secured_identities.items():
            opened = os.fstat(by_path[path])
            if (opened.st_dev, opened.st_ino) != identity:
                return False
    except (KeyError, OSError):
        return False
    return True


def _entry_matches_fd(parent_fd: int, name: str, opened_fd: int) -> bool:
    try:
        entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        opened = os.fstat(opened_fd)
    except OSError:
        return False
    return stat.S_ISDIR(entry.st_mode) and (entry.st_dev, entry.st_ino) == (
        opened.st_dev,
        opened.st_ino,
    )


def _path_identity(path: Path) -> tuple[int, int]:
    path_stat = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(path_stat.st_mode):
        raise OSError("publication path is not a directory")
    return path_stat.st_dev, path_stat.st_ino


def _path_chain_matches(
    identities: Mapping[Path, tuple[int, int]], *, profile_root: Path
) -> bool:
    try:
        profile_resolved = profile_root.resolve(strict=True)
        for path, identity in identities.items():
            if _path_identity(path) != identity:
                return False
            resolved = path.resolve(strict=True)
            if path != profile_root and not resolved.is_relative_to(profile_resolved):
                return False
    except OSError:
        return False
    return True


def _path_matches_identity(path: Path, expected_identity: tuple[int, int]) -> bool:
    try:
        return _path_identity(path) == expected_identity
    except OSError:
        return False


def _remaining_publication_candidate_relpath(
    *,
    posix_guards: bool,
    versions_fd: int,
    staging_fd: int,
    final_name: str,
    final_relpath: str,
) -> str | None:
    if posix_guards:
        if (
            versions_fd >= 0
            and staging_fd >= 0
            and _entry_matches_fd(versions_fd, final_name, staging_fd)
        ):
            return final_relpath
        return None
    return None


def _write_private_publication_path(
    directory: Path, filename: str, data: bytes
) -> None:
    if _safe_relative_parts(filename) != (filename,):
        raise ValueError("visual_identity_candidate_invalid")
    path = directory / filename
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _read_private_publication_path(
    directory: Path, filename: str, max_bytes: int
) -> bytes:
    if _safe_relative_parts(filename) != (filename,):
        raise ValueError("visual_identity_candidate_invalid")
    path = directory / filename
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        return _read_bounded_publication_descriptor(
            descriptor,
            entry_stat=lambda: os.stat(path, follow_symlinks=False),
            max_bytes=max_bytes,
        )
    finally:
        os.close(descriptor)


def _sync_publication_directory(directory: int | Path) -> None:
    descriptor = directory if isinstance(directory, int) else -1
    try:
        if descriptor < 0:
            if os.name == "nt":
                return
            descriptor = os.open(
                directory,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
        os.fsync(descriptor)
    except OSError as error:
        unsupported = {
            errno.EBADF,
            errno.EINVAL,
            getattr(errno, "ENOTSUP", errno.EINVAL),
            getattr(errno, "EOPNOTSUPP", errno.EINVAL),
        }
        if error.errno not in unsupported:
            raise
    finally:
        if not isinstance(directory, int) and descriptor >= 0:
            os.close(descriptor)


def _read_private_publication_file(
    directory_fd: int, filename: str, *, max_bytes: int
) -> bytes:
    if _safe_relative_parts(filename) != (filename,):
        raise ValueError("visual_identity_candidate_invalid")
    descriptor = os.open(
        filename,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        dir_fd=directory_fd,
    )
    try:
        return _read_bounded_publication_descriptor(
            descriptor,
            entry_stat=lambda: os.stat(
                filename, dir_fd=directory_fd, follow_symlinks=False
            ),
            max_bytes=max_bytes,
        )
    finally:
        os.close(descriptor)


def _read_bounded_publication_descriptor(
    descriptor: int,
    *,
    entry_stat: Callable[[], os.stat_result],
    max_bytes: int,
) -> bytes:
    opened = os.fstat(descriptor)
    before = entry_stat()
    identity = (opened.st_dev, opened.st_ino)
    if (
        not stat.S_ISREG(opened.st_mode)
        or opened.st_size > max_bytes
        or (before.st_dev, before.st_ino) != identity
    ):
        raise ValueError("visual_identity_candidate_invalid")
    chunks: list[bytes] = []
    byte_count = 0
    while byte_count <= max_bytes:
        chunk = os.read(descriptor, min(64 * 1024, max_bytes + 1 - byte_count))
        if not chunk:
            break
        chunks.append(chunk)
        byte_count += len(chunk)
    after = entry_stat()
    if byte_count > max_bytes or (after.st_dev, after.st_ino) != identity:
        raise ValueError("visual_identity_candidate_invalid")
    return b"".join(chunks)


def _discard_staging_directory(
    versions_fd: int, staging_name: str, staging_fd: int
) -> bool:
    if not staging_name.startswith(".staging-"):
        return False
    return _discard_pinned_directory(versions_fd, staging_name, staging_fd)


def _discard_pinned_directory(parent_fd: int, entry_name: str, pinned_fd: int) -> bool:
    try:
        if _safe_relative_parts(entry_name) != (entry_name,):
            return False
        entry_stat = os.stat(
            entry_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if pinned_fd < 0 or not stat.S_ISDIR(entry_stat.st_mode):
            return False
        pinned_stat = os.fstat(pinned_fd)
        if (entry_stat.st_dev, entry_stat.st_ino) != (
            pinned_stat.st_dev,
            pinned_stat.st_ino,
        ):
            return False
        for filename in os.listdir(pinned_fd):
            child_stat = os.stat(filename, dir_fd=pinned_fd, follow_symlinks=False)
            if stat.S_ISDIR(child_stat.st_mode):
                return False
            os.unlink(filename, dir_fd=pinned_fd)
        if not _entry_matches_fd(parent_fd, entry_name, pinned_fd):
            return False
        os.rmdir(entry_name, dir_fd=parent_fd)
        return True
    except OSError:
        return False


def _reset_candidate_publication(candidate: VisualIdentityCandidate) -> None:
    with candidate._lock:
        if not candidate._published:
            candidate._publishing = False


_SAMIRA_CARD_NAME = "Samira “Sammy” Vadem"
_SAMIRA_RESOURCE_ROOT = ("assets", "characters", "samira")
_SAMIRA_TOP_LEVEL_FILES = frozenset(
    {
        "ASSET_LICENSE.md",
        "Samira.character.json",
        "Sammy.png",
        "visual_identity_pack.json",
    }
)
_SAMIRA_TEXT_LIMIT = 2 * 1024 * 1024
_SAMIRA_PORTRAIT_LIMIT = 10 * 1024 * 1024
_SAMIRA_SEED_LOCKS: dict[str, threading.Lock] = {}
_SAMIRA_SEED_LOCKS_GUARD = threading.Lock()


def ensure_builtin_samira(
    db: Any,
    package_root: str | Path | None = None,
    user_data_dir: str | Path | None = None,
) -> None:
    """Create the bundled Samira card and pack once without rewriting user state.

    Args:
        db: Initialized profile-local ``CharactersRAGDB``.
        package_root: Injectable package root used by isolated tests.
        user_data_dir: Reserved profile root for the later copy-on-write path.
    """
    del user_data_dir
    state = _samira_seed_preflight(db)
    if state["terminal"]:
        return

    with _samira_seed_lock(db):
        state = _samira_seed_preflight(db)
        if state["terminal"]:
            return

        card = state["card"]
        try:
            card_json, portrait, parsed_card = _load_samira_card(package_root)
        except Exception as exc:  # noqa: BLE001 - startup seed must not prevent boot
            logger.warning("samira_card_seed_failed category={}", type(exc).__name__)
            return

        if card is None:
            parsed_card["name"] = _available_samira_name(db)
            parsed_card["image"] = portrait
            character_id = db.add_character_card(parsed_card)
            if character_id is None:
                logger.warning("samira_card_seed_failed category=insert")
                return
        else:
            character_id = int(card["id"])

        try:
            manifest_data, manifest, loaded_assets = _load_samira_pack(
                package_root,
                card_bytes=len(card_json),
                portrait_bytes=len(portrait),
            )
            from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository

            VisualIdentityRepository(db).activate_pack(
                pack={
                    "title": manifest.title,
                    "description": "Bundled Samira reaction pack.",
                    "default_expression_key": manifest.default_expression_key,
                    "source_kind": "builtin",
                    "source_context": {
                        "source_id": SAMIRA_PACK_ID,
                        "pack_content_sha256": manifest.pack_content_sha256,
                    },
                },
                manifest=manifest_data,
                assets=[
                    {
                        "expression_key": loaded.asset.expression_key,
                        "original_expression_key": loaded.asset.original_label,
                        "display_label": loaded.asset.display_label,
                        "source_filename": PurePosixPath(
                            loaded.asset.storage_relpath
                        ).name,
                        "storage_relpath": loaded.asset.storage_relpath,
                        "content_type": loaded.asset.content_type,
                        "bytes": loaded.asset.bytes,
                        "sha256": loaded.asset.sha256,
                        "width": loaded.asset.width,
                        "height": loaded.asset.height,
                        "source_context": _asset_generation_context(
                            manifest_data, loaded.asset.original_label
                        ),
                        "is_animated": loaded.asset.is_animated,
                        "frame_count": loaded.asset.frame_count,
                        "duration_ms": loaded.asset.duration_ms,
                    }
                    for loaded in loaded_assets
                ],
                actor_kind="character",
                actor_id=character_id,
            )
        except Exception as exc:  # noqa: BLE001 - card remains usable on pack failure
            logger.warning(
                "samira_pack_activation_failed category={}", type(exc).__name__
            )


def _samira_seed_lock(db: Any) -> threading.Lock:
    db_path = str(getattr(db, "db_path_str", ":memory:"))
    key = f"memory:{id(db)}" if db_path == ":memory:" else str(Path(db_path).resolve())
    with _SAMIRA_SEED_LOCKS_GUARD:
        return _SAMIRA_SEED_LOCKS.setdefault(key, threading.Lock())


def _samira_seed_preflight(db: Any) -> dict[str, Any]:
    card = _find_builtin_samira_card(db)
    pack = _find_builtin_samira_pack(db)
    if card is not None and int(card["deleted"]):
        return {"terminal": True, "card": card}
    if card is not None:
        bindings = [
            dict(row)
            for row in db.execute_query(
                """SELECT * FROM visual_identity_bindings
                     WHERE owner_user_id = 0 AND actor_kind = 'character'
                       AND actor_id = ? ORDER BY id""",
                (str(card["id"]),),
            ).fetchall()
        ]
        if any(binding["status"] == "deleted" for binding in bindings):
            return {"terminal": True, "card": card}
        active = next(
            (binding for binding in bindings if binding["status"] == "active"), None
        )
        if active is not None and _binding_is_terminal(db, active):
            return {"terminal": True, "card": card}
    if pack is not None:
        return {"terminal": True, "card": card}
    return {"terminal": False, "card": card}


_SAMIRA_CARD_BY_BUILTIN_ID_SQL = """
    SELECT id, name, extensions, deleted
      FROM character_cards
     WHERE CASE WHEN json_valid(extensions)
                THEN json_extract(extensions, '$."tldw/builtin_id"')
           END = 'samira'
     ORDER BY id
     LIMIT 1
"""


def _find_builtin_samira_card(db: Any) -> dict[str, Any] | None:
    """Return the bundled Samira card row, or ``None``.

    Runs on every boot (``ensure_builtin_samira``'s preflight), so it asks
    SQLite the question instead of reading the whole table into Python and
    parsing every card's ``extensions`` JSON (TASK-21111(d)): measured 3.0x
    faster at 2,000 cards (4.29 ms -> 1.43 ms) and 2.8x at 100 (0.185 ms ->
    0.067 ms).

    Two details make the SQL agree with the Python loop it replaced on the
    rows the loop tolerated:

    * ``json_extract`` RAISES ``malformed JSON`` on an unparseable
      ``extensions`` value (verified on SQLite 3.49.1), where the loop simply
      skipped that row -- so the ``json_valid`` guard is load-bearing, not
      decoration, and is written as a ``CASE`` so the language, not the query
      planner, guarantees it is evaluated first. ``json_valid`` also rejects
      the ``NaN``/``Infinity`` constants ``_reject_json_constant`` rejects,
      and ``json_extract`` yields NULL (no match) for non-object JSON,
      matching the loop's ``isinstance(..., dict)`` check.
    * ``ORDER BY id LIMIT 1`` preserves "lowest id wins".

    Falls back to the historical Python scan only when the SQLite build has
    no JSON1 functions. The fallback is deliberately NOT a catch-all: an
    every-boot silent full scan is exactly the cost this function exists to
    remove, so any other failure propagates (``seed_builtin_content`` already
    contains it) rather than hiding behind a slow path.
    """
    try:
        row = db.execute_query(_SAMIRA_CARD_BY_BUILTIN_ID_SQL).fetchone()
    except Exception as exc:  # noqa: BLE001 - inspected and re-raised below
        if "no such function" not in str(exc).lower():
            raise
        logger.debug("samira_card_lookup_fallback category={}", type(exc).__name__)
        return _find_builtin_samira_card_by_scan(db)
    return dict(row) if row is not None else None


def _find_builtin_samira_card_by_scan(db: Any) -> dict[str, Any] | None:
    """Full-table fallback for SQLite builds without the JSON1 extension."""
    rows = db.execute_query(
        "SELECT id, name, extensions, deleted FROM character_cards ORDER BY id"
    ).fetchall()
    for row in rows:
        candidate = dict(row)
        try:
            extensions = json.loads(
                candidate.get("extensions") or "{}",
                parse_constant=_reject_json_constant,
            )
        except (TypeError, ValueError):
            continue
        if (
            isinstance(extensions, dict)
            and extensions.get("tldw/builtin_id") == "samira"
        ):
            return candidate
    return None


def _find_builtin_samira_pack(db: Any) -> dict[str, Any] | None:
    rows = db.execute_query(
        """SELECT * FROM visual_identity_packs
             WHERE owner_user_id = 0 AND source_kind = 'builtin' ORDER BY id"""
    ).fetchall()
    for row in rows:
        candidate = dict(row)
        try:
            context = json.loads(
                candidate.get("source_context_json") or "{}",
                parse_constant=_reject_json_constant,
            )
        except (TypeError, ValueError):
            continue
        if isinstance(context, dict) and context.get("source_id") == SAMIRA_PACK_ID:
            return candidate
    return None


def _binding_is_terminal(db: Any, binding: Mapping[str, Any]) -> bool:
    row = db.execute_query(
        """SELECT p.*, v.pack_id AS version_pack_id,
                  (SELECT COUNT(*) FROM visual_identity_assets a
                    WHERE a.pack_version_id = v.id AND a.deleted = 0) AS asset_count,
                  (SELECT COUNT(*) FROM visual_identity_assets a
                    WHERE a.pack_version_id = v.id AND a.deleted = 0
                      AND a.expression_key = v.default_expression_key) AS default_count
             FROM visual_identity_packs p
             JOIN visual_identity_pack_versions v ON v.id = ?
            WHERE p.id = ? AND p.owner_user_id = 0 AND p.status = 'active'""",
        (binding["active_version_id"], binding["pack_id"]),
    ).fetchone()
    if row is None:
        return False
    pack = dict(row)
    if int(pack["version_pack_id"]) != int(pack["id"]):
        return False
    if int(pack["asset_count"]) < 1 or int(pack["default_count"]) != 1:
        return False
    if pack["source_kind"] == "manual":
        return True
    if pack["source_kind"] != "builtin" or int(pack["asset_count"]) != len(
        SAMIRA_REACTION_LABELS
    ):
        return False
    try:
        context = json.loads(
            pack["source_context_json"], parse_constant=_reject_json_constant
        )
    except (TypeError, ValueError):
        return False
    return isinstance(context, dict) and context.get("source_id") == SAMIRA_PACK_ID


def _available_samira_name(db: Any) -> str:
    occupied = {
        str(row[0])
        for row in db.execute_query("SELECT name FROM character_cards").fetchall()
    }
    if _SAMIRA_CARD_NAME not in occupied:
        return _SAMIRA_CARD_NAME
    fallback = f"{_SAMIRA_CARD_NAME} (Built-in)"
    if fallback not in occupied:
        return fallback
    suffix = 2
    while f"{fallback} {suffix}" in occupied:
        suffix += 1
    return f"{fallback} {suffix}"


def _load_samira_card(
    package_root: str | Path | None,
) -> tuple[bytes, bytes, dict[str, Any]]:
    card_bytes = _read_samira_resource(
        package_root, "Samira.character.json", max_bytes=_SAMIRA_TEXT_LIMIT
    )
    portrait = _read_samira_resource(
        package_root, "Sammy.png", max_bytes=_SAMIRA_PORTRAIT_LIMIT
    )
    card = _strict_json_object(card_bytes)
    data = card.get("data")
    if (
        card.get("spec") != "chara_card_v2"
        or card.get("spec_version") != "2.0"
        or not isinstance(data, dict)
        or not isinstance(data.get("extensions"), dict)
        or data["extensions"].get("tldw/builtin_id") != "samira"
        or data["extensions"].get("tldw/visual_identity_pack_id") != SAMIRA_PACK_ID
    ):
        raise ValueError("samira_card_invalid")

    from tldw_chatbook.Character_Chat.Character_Chat_Lib import (
        extract_json_from_image_file,
        parse_v2_card,
    )

    embedded = extract_json_from_image_file(BytesIO(portrait))
    if embedded is None or _strict_json_object(embedded) != card:
        raise ValueError("samira_card_mismatch")
    parsed = parse_v2_card(card)
    if parsed is None:
        raise ValueError("samira_card_invalid")
    return card_bytes, portrait, parsed


def _load_samira_pack(
    package_root: str | Path | None,
    *,
    card_bytes: int,
    portrait_bytes: int,
) -> tuple[
    dict[str, Any],
    VisualIdentityManifest,
    tuple[LoadedVisualIdentityAsset, ...],
]:
    _validate_samira_resource_inventory(package_root)
    manifest_raw = _read_samira_resource(
        package_root, "visual_identity_pack.json", max_bytes=_SAMIRA_TEXT_LIMIT
    )
    license_raw = _read_samira_resource(
        package_root, "ASSET_LICENSE.md", max_bytes=_SAMIRA_TEXT_LIMIT
    )
    manifest_data = _strict_json_object(manifest_raw)
    metadata_manifest = validate_visual_identity_manifest(
        manifest_data,
        require_samira_bundle=True,
        directory_bytes=0,
    )

    asset_bytes: dict[str, bytes] = {}
    directory_bytes = card_bytes + portrait_bytes + len(manifest_raw) + len(license_raw)
    for asset in metadata_manifest.assets:
        parts = _safe_relative_parts(asset.storage_relpath)
        if parts[:3] != ("characters", "samira", "expressions"):
            raise ValueError("samira_manifest_invalid")
        data = _read_samira_resource(
            package_root,
            "/".join(parts[2:]),
            max_bytes=SAMIRA_MAX_REACTION_BYTES + 1,
        )
        asset_bytes[asset.storage_relpath] = data
        directory_bytes += len(data)

    manifest = parse_visual_identity_manifest_json(
        manifest_raw,
        require_samira_bundle=True,
        directory_bytes=directory_bytes,
    )
    loaded: list[LoadedVisualIdentityAsset] = []
    decoded_pixels = 0
    for asset in manifest.assets:
        data = asset_bytes[asset.storage_relpath]
        if len(data) != asset.bytes:
            raise ValueError("visual_identity_asset_size_mismatch")
        if hashlib.sha256(data).hexdigest() != asset.sha256:
            raise ValueError("visual_identity_asset_sha256_mismatch")
        item = LoadedVisualIdentityAsset(asset=asset, data=data)
        decoded_pixels += _validate_image_bytes(
            item, decoded_pixels_before=decoded_pixels
        )
        loaded.append(item)
    return manifest_data, manifest, tuple(loaded)


def _read_samira_resource(
    package_root: str | Path | None,
    relative_path: str,
    *,
    max_bytes: int,
) -> bytes:
    parts = _safe_relative_parts(relative_path)
    try:
        root = (
            Path(package_root)
            if package_root is not None
            else resources.files("tldw_chatbook")
        )
        candidate = root.joinpath(*_SAMIRA_RESOURCE_ROOT, *parts)
        with candidate.open("rb") as stream:
            data = stream.read(max_bytes + 1)
    except (OSError, RuntimeError, TypeError, AttributeError):
        raise ValueError("samira_resource_unavailable") from None
    if not isinstance(data, bytes) or len(data) > max_bytes:
        raise ValueError("samira_resource_invalid")
    return data


def _validate_samira_resource_inventory(package_root: str | Path | None) -> None:
    try:
        root = (
            Path(package_root)
            if package_root is not None
            else resources.files("tldw_chatbook")
        ).joinpath(*_SAMIRA_RESOURCE_ROOT)
        top_level = {entry.name for entry in root.iterdir()}
        expressions = {entry.name for entry in root.joinpath("expressions").iterdir()}
    except (OSError, RuntimeError, TypeError, AttributeError):
        raise ValueError("samira_resource_unavailable") from None
    if top_level != _SAMIRA_TOP_LEVEL_FILES | {"expressions"}:
        raise ValueError("samira_resource_inventory_invalid")
    if expressions != {f"{label}.webp" for label in SAMIRA_REACTION_LABELS}:
        raise ValueError("samira_resource_inventory_invalid")


def _strict_json_object(raw: bytes | str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8", errors="strict") if isinstance(raw, bytes) else raw
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError):
        raise ValueError("samira_json_invalid") from None
    if not isinstance(value, dict):  # noqa: TRY004 - stable validation category
        raise ValueError("samira_json_invalid")
    return value


def _asset_generation_context(
    manifest: Mapping[str, Any], original_label: str
) -> dict[str, Any]:
    assets = manifest.get("assets")
    if not isinstance(assets, list):
        return {}
    for asset in assets:
        if isinstance(asset, dict) and asset.get("original_label") == original_label:
            generation = asset.get("generation")
            return dict(generation) if isinstance(generation, dict) else {}
    return {}
