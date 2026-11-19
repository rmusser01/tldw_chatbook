"""Existing Personas thumbnail helpers shared by the screen's public seams."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from textual.css.query import QueryError

from tldw_chatbook.Actor_Packs.contracts import ActorPackValidationError
from tldw_chatbook.Character_Chat.expression_generation import (
    canonical_visual_identity_reactions,
)
from tldw_chatbook.Character_Chat.visual_identity import (
    CANONICAL_EXPRESSION_SLOTS,
    VisualIdentityCandidate,
    VisualIdentityResolution,
    display_label_for_expression_key,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
    VisualIdentityAssetMetadata,
    VisualIdentityPackMetadata,
)
from tldw_chatbook.Widgets.Persona_Widgets.personas_state import (
    _ActorPackPortraitChoice,
    _VisualIdentityPreviewSnapshot,
)

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_image_view import ConsoleImageRenderCache
from tldw_chatbook.Character_Chat.character_avatar import (
    AVATAR_IMAGE_SUFFIX_COPY as PERSONAS_AVATAR_IMAGE_SUFFIX_COPY,
)
from tldw_chatbook.Character_Chat.character_avatar import (
    AVATAR_IMAGE_SUFFIXES as PERSONAS_AVATAR_IMAGE_SUFFIXES,
)
from tldw_chatbook.Character_Chat.character_avatar import (
    AVATAR_MAX_BYTES as PERSONAS_AVATAR_MAX_BYTES,
)
from tldw_chatbook.Character_Chat.character_avatar import (
    AVATAR_MAX_SIZE_COPY as PERSONAS_AVATAR_MAX_SIZE_COPY,
)
from tldw_chatbook.Chat.console_expression_state import EXPRESSION_IMAGE_STATES
from tldw_chatbook.Utils.path_validation import validate_path_simple
from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
    PersonasCharacterEditorWidget,
)

AVATAR_THUMB_COLS = 24
AVATAR_THUMB_LINES = 10


def _read_avatar_image_bytes(self, path: str) -> bytes:
    candidate = validate_path_simple(path, require_exists=True)
    if not candidate.is_file():
        raise ValueError("Choose an existing avatar image file.")
    if candidate.suffix.lower() not in PERSONAS_AVATAR_IMAGE_SUFFIXES:
        raise ValueError(
            f"Unsupported avatar image type. Use {PERSONAS_AVATAR_IMAGE_SUFFIX_COPY}."
        )
    if candidate.stat().st_size > PERSONAS_AVATAR_MAX_BYTES:
        raise ValueError(
            f"Avatar image must be {PERSONAS_AVATAR_MAX_SIZE_COPY} or smaller."
        )
    data = candidate.read_bytes()
    if not data:
        raise ValueError("Avatar image file is empty.")
    return data


def _fit_avatar_cell_size(pixel_width: int, pixel_height: int) -> tuple[int, int]:
    """Fit a PIL image's pixel size into the avatar thumb box, in cells.

    Terminal cells are roughly twice as tall (in pixels) as they are
    wide, so the image's aspect ratio is first converted from pixels to
    "cell units" (halving the height) before fitting it into the
    ``AVATAR_THUMB_COLS`` x ``AVATAR_THUMB_LINES`` box. Both returned
    dimensions are explicit ints >= 1 - leaving either as "auto" is what
    reintroduced the 0-size ``ValueError`` this rendering path already
    works around (see the caller's comment).

    Args:
        pixel_width: Source image width in pixels.
        pixel_height: Source image height in pixels.

    Returns:
        ``(width_cells, height_cells)``, each clamped to
        ``[1, AVATAR_THUMB_COLS]`` / ``[1, AVATAR_THUMB_LINES]``.
    """
    if pixel_width <= 0 or pixel_height <= 0:
        return AVATAR_THUMB_COLS, AVATAR_THUMB_LINES
    # Aspect ratio expressed in cell units (width-cells : height-cells).
    cell_aspect = pixel_width / (pixel_height / 2)
    box_aspect = AVATAR_THUMB_COLS / AVATAR_THUMB_LINES
    if cell_aspect >= box_aspect:
        # Image is relatively wider than the box - fit to width.
        w_cells = AVATAR_THUMB_COLS
        h_cells = max(1, round(AVATAR_THUMB_COLS / cell_aspect))
    else:
        # Image is relatively taller than the box - fit to height.
        h_cells = AVATAR_THUMB_LINES
        w_cells = max(1, round(AVATAR_THUMB_LINES * cell_aspect))
    w_cells = max(1, min(AVATAR_THUMB_COLS, w_cells))
    h_cells = max(1, min(AVATAR_THUMB_LINES, h_cells))
    return w_cells, h_cells


def _build_avatar_pixels(cache: ConsoleImageRenderCache, cache_key: str):
    """Build a quadrant-mosaic renderable sized to the avatar thumb box.

    ``cache.get_pixels`` thumbnails to the 80x40 chat-transcript box,
    which is far larger than the character editor's compact avatar
    preview; reusing it produces an oversized grid that Rich does not
    reflow, so the small thumb container just crops it to a top-left
    sliver. This instead pulls the cached full-size PIL image and bakes
    it to the avatar box via the quadrant mosaic (2x2 subpixels per
    cell -- double the horizontal detail of the old half-block Pixels
    renderer, same universal Block Elements font coverage).

    Args:
        cache: The render cache holding the decoded avatar image.
        cache_key: The per-session cache key ``prepare`` was called with.

    Returns:
        A mosaic ``Text`` renderable fitted to the avatar box, or
        ``None`` when the image is not (or no longer) cached.
    """
    from ...Utils.mosaic_render import mosaic_from_image

    pil = cache.get_pil(cache_key)
    if pil is None:
        return None
    return mosaic_from_image(pil, AVATAR_THUMB_COLS, AVATAR_THUMB_LINES, fit="cover")


async def _render_all_character_editor_thumbnails(
    self, character_id: int | None
) -> None:
    """Render the avatar thumbnail plus all 3 expression-state thumbnails.

    Called whenever the character editor session's identity changes: a
    new create/edit session opens, or a save assigns a brand-new
    character its first id (Roleplay P3d-1 Task 4). ``character_id``
    gates the expression slots - ``None`` (a still-unsaved character)
    clears them instead of reading the DB, since there is no row to read.

    Args:
        character_id: The loaded character's row id, or ``None`` when
            the editor session has not been saved yet.
    """
    await self._render_character_editor_avatar()
    try:
        editor = self.query_one(PersonasCharacterEditorWidget)
    except QueryError:
        return
    if character_id is None:
        for state in EXPRESSION_IMAGE_STATES:
            editor.set_expression_thumbnail(state, None)
        return
    for state in EXPRESSION_IMAGE_STATES:
        await self._render_character_expression_slot(character_id, state)


def _unbound_persona_shared_visual_identity_metadata() -> (
    VisualIdentityPackMetadata
):
    """Return canonical path-free slots for creating a Persona binding."""

    return VisualIdentityPackMetadata(
        binding_id=0,
        pack_id=0,
        pack_version_id=0,
        title="Shared Visual Identity reactions",
        source_kind="unbound",
        default_expression_key="neutral",
        assets=tuple(
            VisualIdentityAssetMetadata(
                asset_id=-index,
                expression_key=key,
                original_label=key,
                display_label=display_label_for_expression_key(key),
                content_type="",
                is_animated=False,
            )
            for index, key in enumerate(CANONICAL_EXPRESSION_SLOTS, start=1)
        ),
    )

def _visual_identity_pack_metadata(
    graph: dict[str, Any] | None,
) -> VisualIdentityPackMetadata | None:
    """Build the complete path-free identity used for live graph checks."""

    if graph is None:
        return None

    pack_row = graph["pack"]
    version_row = graph["version"]
    return VisualIdentityPackMetadata(
        binding_id=int(graph["binding"]["id"]),
        pack_id=int(pack_row["id"]),
        pack_version_id=int(version_row["id"]),
        title=str(pack_row["title"]),
        source_kind=str(pack_row["source_kind"]),
        default_expression_key=str(version_row["default_expression_key"]),
        assets=tuple(
            VisualIdentityAssetMetadata(
                asset_id=int(asset["id"]),
                expression_key=str(asset["expression_key"]),
                original_label=str(asset["original_expression_key"]),
                display_label=str(asset["display_label"]),
                content_type=str(asset["content_type"]),
                is_animated=bool(asset["is_animated"]),
            )
            for asset in graph["assets"]
        ),
    )

def _visual_identity_graph_matches_snapshot(
    snapshot: _VisualIdentityPreviewSnapshot,
    graph: dict[str, Any] | None,
) -> bool:
    """Check the live binding graph without reading or decoding asset bytes."""

    if graph is None:
        return False
    binding = graph["binding"]
    pack = graph["pack"]
    version = graph["version"]
    asset = next(
        (
            row
            for row in graph["assets"]
            if str(row["expression_key"]) == snapshot.asset.expression_key
        ),
        None,
    )
    return bool(
        asset is not None
        and int(binding["id"]) == snapshot.binding_id
        and int(pack["id"]) == snapshot.pack_id
        and int(version["id"]) == snapshot.pack_version_id
        and str(pack["source_kind"]) == snapshot.source_kind
        and int(asset["id"]) == snapshot.asset.asset_id
        and str(asset["content_type"]) == snapshot.asset.content_type
        and bool(asset["is_animated"]) == snapshot.asset.is_animated
    )

def _visual_identity_resolution_identity(
    resolution: VisualIdentityResolution,
) -> tuple[object, ...]:
    """Return every source field that can distinguish selected pixels."""

    return (
        resolution.actor_kind,
        resolution.actor_id,
        resolution.requested_expression_key,
        resolution.manual_expression_key,
        resolution.resolved_expression_key,
        resolution.pack_id,
        resolution.pack_version_id,
        resolution.asset_id,
        resolution.expression_id,
        resolution.storage_source,
        resolution.storage_relpath,
        resolution.content_type,
        resolution.is_animated,
        resolution.resolution_source,
        resolution.fallback_reason,
        resolution.cache_identity,
    )

def _visual_identity_direction(
    candidate: VisualIdentityCandidate, asset: VisualIdentityAssetMetadata
) -> str:
    """Read the retained asset's generation direction, with a stable fallback."""

    row = next(
        (
            stored
            for stored in candidate.assets
            if str(stored.get("expression_key")) == asset.expression_key
        ),
        {},
    )
    context = row.get("source_context_json", row.get("source_context", {}))
    if isinstance(context, str):
        try:
            context = json.loads(context)
        except (TypeError, ValueError, json.JSONDecodeError):
            context = {}
    direction = (
        context.get("visual_direction") if isinstance(context, dict) else None
    )
    return str(direction or asset.display_label)

def _canonical_visual_identity_assets() -> tuple[VisualIdentityAssetMetadata, ...]:
    """Build path-free metadata for every approved reaction row."""

    return tuple(
        VisualIdentityAssetMetadata(
            asset_id=-(index + 1),
            expression_key=reaction.expression_key,
            original_label=reaction.original_label,
            display_label=reaction.display_label,
            content_type="image/png",
            is_animated=False,
        )
        for index, reaction in enumerate(canonical_visual_identity_reactions())
    )


def _actor_pack_portrait_name(data: bytes) -> str:
    """Infer the only supported bounded raster suffix without exposing a path."""

    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "portrait.png"
    if data.startswith(b"\xff\xd8\xff"):
        return "portrait.jpg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "portrait.gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "portrait.webp"
    raise ValueError("actor_pack_portrait_invalid")

def _eligible_actor_pack_portraits(
    records: list[object],
    *,
    validate_actor_portrait: Callable[[str, bytes], Any],
) -> tuple[_ActorPackPortraitChoice, ...]:
    choices: list[_ActorPackPortraitChoice] = []
    for raw in records:
        if not isinstance(raw, Mapping):
            continue
        character_id = raw.get("id")
        revision = raw.get("version")
        name = raw.get("name")
        portrait = raw.get("image")
        if (
            type(character_id) is not int
            or character_id < 1
            or type(revision) is not int
            or revision < 1
            or type(name) is not str
            or not name.strip()
            or type(portrait) is not bytes
        ):
            continue
        try:
            validate_actor_portrait(_actor_pack_portrait_name(portrait), portrait)
        except (ActorPackValidationError, ValueError):
            continue
        choices.append(
            _ActorPackPortraitChoice(
                character_id=character_id,
                name=name.strip(),
                revision=revision,
                sha256=hashlib.sha256(portrait).hexdigest(),
            )
        )
    return tuple(
        sorted(choices, key=lambda choice: (choice.name, choice.character_id))
    )
