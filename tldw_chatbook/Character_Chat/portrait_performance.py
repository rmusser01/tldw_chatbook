"""Bounded profile-local portrait poses and canonical persisted source identity."""

from __future__ import annotations

import hashlib
import re
import warnings
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from io import BytesIO
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from sqlite3 import Cursor

    from tldw_chatbook.Character_Chat.visual_identity import VisualIdentityResolution
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    from .portrait_compositing import PortraitCompositionRecipe

FrameRole = Literal["rest", "small", "open", "blink_rest", "blink_small", "blink_open"]
FRAME_ROLES: tuple[FrameRole, ...] = (
    "rest",
    "small",
    "open",
    "blink_rest",
    "blink_small",
    "blink_open",
)
MAX_FRAME_PIXELS = 1_048_576
MAX_FRAME_BYTES = 4 * 1024 * 1024
MAX_SET_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class PerformanceFrame:
    """One normalized, independently validated still pose."""

    role: FrameRole
    png: bytes = field(repr=False)
    width: int
    height: int
    sha256: str


@dataclass(frozen=True, slots=True)
class PerformanceSource:
    """Stable content identity, excluding session/profile authority and paths."""

    character_id: int
    expression_key: str
    source_sha256: str
    source_kind: Literal["portrait", "legacy_expression", "visual_identity"]
    source_ref: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PortraitPerformanceRecord:
    """Current attachment revision, including inactive removal tombstones."""

    source: PerformanceSource
    revision: int
    active: bool
    frames: tuple[PerformanceFrame, ...] = field(repr=False)
    composition_recipe: PortraitCompositionRecipe | None = None


class PerformanceConflict(ValueError):
    """The persisted source or attachment revision changed before publication."""


def normalize_performance_frame(role: FrameRole, raw: bytes) -> PerformanceFrame:
    """Decode a bounded still PNG/JPEG/WebP into a metadata-free RGBA PNG.

    Raises:
        ValueError: For unsupported, animated, invalid or over-budget artwork.
    """
    from PIL import Image

    from tldw_chatbook.Character_Chat import visual_identity as visual
    from tldw_chatbook.Chat.character_expression_playback import _adjust_budget

    if role not in FRAME_ROLES or type(raw) is not bytes or not raw:
        raise ValueError("portrait_performance_frame_invalid")
    if len(raw) > min(MAX_FRAME_BYTES, visual.MAX_EXPRESSION_ASSET_BYTES):
        raise ValueError("portrait_performance_frame_bytes_exceeded")
    reserved = 0
    try:
        with ExitStack() as stack, warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            source = stack.enter_context(BytesIO(raw))
            image = Image.open(source)
            stack.callback(image.close)
            width, height = image.size
            if (
                image.format not in {"PNG", "JPEG", "WEBP"}
                or getattr(image, "n_frames", 1) != 1
                or getattr(image, "is_animated", False)
                or width < 1
                or height < 1
                or width * height
                > min(MAX_FRAME_PIXELS, visual.MAX_EXPRESSION_ASSET_DECODED_PIXELS)
                or max(width, height) > visual.MAX_EXPRESSION_IMAGE_DIMENSION
            ):
                raise ValueError("portrait_performance_image_limits_exceeded")
            scratch = width * height * 12
            _adjust_budget(scratch)
            reserved = scratch
            image.load()
            rgba = image.convert("RGBA")
            stack.callback(rgba.close)
            rgba.info.clear()
            output = stack.enter_context(BytesIO())
            rgba.save(output, format="PNG")
            data = output.getvalue()
            if len(data) > min(MAX_FRAME_BYTES, visual.MAX_EXPRESSION_ASSET_BYTES):
                raise ValueError("portrait_performance_frame_bytes_exceeded")
            check = Image.open(stack.enter_context(BytesIO(data)))
            stack.callback(check.close)
            check.verify()
            return PerformanceFrame(
                role, data, width, height, hashlib.sha256(data).hexdigest()
            )
    except (
        OSError,
        SyntaxError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ValueError("portrait_performance_image_invalid") from exc
    finally:
        if reserved:
            _adjust_budget(-reserved)


def normalize_performance_frames(
    frames: Mapping[FrameRole, bytes],
) -> tuple[PerformanceFrame, ...]:
    """Validate exactly six aligned stills without cropping or stretching."""
    if not isinstance(frames, Mapping) or set(frames) != set(FRAME_ROLES):
        raise ValueError("portrait_performance_requires_six_roles")
    normalized = []
    total = 0
    size = None
    for role in FRAME_ROLES:
        frame = normalize_performance_frame(role, frames[role])
        total += len(frame.png)
        if size is not None and size != (frame.width, frame.height):
            raise ValueError("portrait_performance_geometry_mismatch")
        if total > MAX_SET_BYTES:
            raise ValueError("portrait_performance_set_bytes_exceeded")
        size = (frame.width, frame.height)
        normalized.append(frame)
    return tuple(normalized)


def source_from_resolution(resolution: VisualIdentityResolution) -> PerformanceSource:
    """Canonicalize current/historical resolutions of the same actual artwork."""
    if resolution.actor_kind != "character" or not resolution.image_bytes:
        raise PerformanceConflict("portrait_performance_source_unavailable")
    character_id = int(resolution.actor_id)
    if character_id <= 0:
        raise ValueError("portrait_performance_character_invalid")
    if resolution.asset_id is not None:
        kind = "visual_identity"
        refs = (resolution.pack_id, resolution.pack_version_id, resolution.asset_id)
    elif resolution.expression_id is not None:
        kind = "legacy_expression"
        refs = (resolution.expression_id,)
    else:
        kind = "portrait"
        refs = ()
    if any(type(ref) is not int or ref <= 0 for ref in refs):
        raise ValueError("portrait_performance_source_reference_invalid")
    return PerformanceSource(
        character_id,
        "neutral" if kind == "portrait" else resolution.resolved_expression_key,
        hashlib.sha256(resolution.image_bytes).hexdigest(),
        kind,
        refs,
    )


def capture_performance_source(
    db: CharactersRAGDB,
    *,
    character_id: int,
    requested_state: str,
    manual_expression_key: str | None = None,
) -> tuple[PerformanceSource, bytes]:
    """Resolve only a saved local character's persisted active artwork."""
    from tldw_chatbook.Character_Chat.visual_identity import resolve_visual_identity

    if type(character_id) is not int or character_id <= 0:
        raise ValueError("portrait_performance_character_invalid")
    resolution = resolve_visual_identity(
        db,
        actor_kind="character",
        actor_id=character_id,
        requested_state=requested_state,
        manual_expression_key=manual_expression_key,
    )
    source = source_from_resolution(resolution)
    return source, resolution.image_bytes


def _source_stamp(db: CharactersRAGDB, source: PerformanceSource) -> tuple:
    """Fence selection metadata without opening profile asset files in a SQL scope."""
    from tldw_chatbook.Character_Chat.visual_identity import MAX_EXPRESSION_ASSET_BYTES

    graph = db.execute_query(
        """SELECT b.id, b.version, b.pack_id, b.active_version_id,
                  p.status, p.source_kind, p.active_version_id, v.default_expression_key,
                  a.id, a.expression_key, a.sha256, a.storage_relpath,
                  a.bytes, a.width, a.height, a.deleted
           FROM visual_identity_bindings b
           LEFT JOIN visual_identity_packs p ON p.id=b.pack_id AND p.owner_user_id=0
           LEFT JOIN visual_identity_pack_versions v ON v.id=b.active_version_id AND v.owner_user_id=0
           LEFT JOIN visual_identity_assets a ON a.pack_id=b.pack_id
                AND a.pack_version_id=b.active_version_id AND a.owner_user_id=0
                AND a.expression_key IN (?, 'neutral', v.default_expression_key)
           WHERE b.owner_user_id=0 AND b.actor_kind='character'
                AND b.actor_id=? AND b.status='active' ORDER BY b.id, a.id""",
        (source.expression_key, str(source.character_id)),
    ).fetchall()
    if source.source_kind == "visual_identity":
        selected = db.execute_query(
            "SELECT sha256, deleted FROM visual_identity_assets WHERE id=? AND pack_id=? AND pack_version_id=?",
            (source.source_ref[2], source.source_ref[0], source.source_ref[1]),
        ).fetchone()
        digest = selected[0] if selected is not None and not selected[1] else None
    else:
        if source.source_kind == "portrait":
            row = db.execute_query(
                "SELECT CASE WHEN length(image)<=? THEN image END FROM character_cards WHERE id=? AND deleted=0",
                (MAX_EXPRESSION_ASSET_BYTES, source.character_id),
            ).fetchone()
        else:
            row = db.execute_query(
                "SELECT CASE WHEN length(image)<=? THEN image END FROM character_expression_images WHERE id=? AND character_id=? AND deleted=0",
                (MAX_EXPRESSION_ASSET_BYTES, source.source_ref[0], source.character_id),
            ).fetchone()
        data = None if row is None else row[0]
        digest = hashlib.sha256(data).hexdigest() if isinstance(data, bytes) else None
    owner = db.execute_query(
        "SELECT id FROM character_cards WHERE id=? AND deleted=0",
        (source.character_id,),
    ).fetchone()
    return owner is not None, digest, tuple(tuple(row) for row in graph)


def _validate_source(source: PerformanceSource) -> None:
    from tldw_chatbook.Character_Chat.visual_identity import normalize_expression_key

    if (
        not isinstance(source, PerformanceSource)
        or type(source.character_id) is not int
        or source.character_id <= 0
        or type(source.expression_key) is not str
        or normalize_expression_key(source.expression_key) != source.expression_key
        or not isinstance(source.source_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", source.source_sha256) is None
        or type(source.source_kind) is not str
        or source.source_kind
        not in {"portrait", "legacy_expression", "visual_identity"}
        or type(source.source_ref) is not tuple
        or len(source.source_ref)
        != {"portrait": 0, "legacy_expression": 1, "visual_identity": 3}.get(
            source.source_kind
        )
        or any(type(ref) is not int or ref <= 0 for ref in source.source_ref)
    ):
        raise ValueError("portrait_performance_source_invalid")


def _preflight_source(db: CharactersRAGDB, source: PerformanceSource) -> tuple:
    _validate_source(source)
    before = _source_stamp(db, source)
    state = {"custom:speaking": "speaking", "custom:error": "error"}.get(
        source.expression_key, source.expression_key
    )
    current, _ = capture_performance_source(
        db,
        character_id=source.character_id,
        requested_state=state,
        manual_expression_key=source.expression_key,
    )
    after = _source_stamp(db, source)
    if (
        current != source
        or before != after
        or not after[0]
        or after[1] != source.source_sha256
    ):
        raise PerformanceConflict("portrait_performance_source_changed")
    return after


@contextmanager
def _performance_write(db: CharactersRAGDB) -> Iterator[Cursor]:
    """Keep a failed replacement atomic even when its caller catches it in a transaction."""
    with db.transaction(immediate=True) as cursor:
        cursor.execute("SAVEPOINT portrait_performance_write")
        try:
            yield cursor
        except BaseException:
            cursor.execute("ROLLBACK TO SAVEPOINT portrait_performance_write")
            raise
        finally:
            cursor.execute("RELEASE SAVEPOINT portrait_performance_write")
