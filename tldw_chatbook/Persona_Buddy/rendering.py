"""Bounded, Textual-independent Persona Buddy raster preparation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from io import BytesIO

from PIL import Image
from rich_pixels import Pixels

from tldw_chatbook.Persona_Visual.contracts import MAX_ASSET_DIMENSION
from tldw_chatbook.Persona_Visual.repository import PersonaVisualIdentity
from tldw_chatbook.Persona_Visual.runtime import (
    PersonaVisualCacheAsset,
    PersonaVisualCacheIdentity,
    PersonaVisualPortrait,
    PersonaVisualResolvedFrame,
)


PERSONA_BUDDY_FRAME_UNAVAILABLE = "persona_buddy_frame_unavailable"
MAX_PERSONA_BUDDY_PREPARED_FRAMES = 128
MAX_PERSONA_BUDDY_PREPARED_CELLS = 1_048_576
_MAX_FRAME_BYTES = 100 * 1024 * 1024
_MAX_COLS = 256
_MAX_LINES = 128


class PersonaBuddyFrameError(ValueError):
    """A fixed, path-free Buddy frame preparation failure."""

    __slots__ = ("category",)

    def __init__(self) -> None:
        self.category = PERSONA_BUDDY_FRAME_UNAVAILABLE
        super().__init__(self.category)


@dataclass(frozen=True, slots=True)
class PersonaBuddyPreparedFrame:
    """One bounded painted frame with complete immutable cache identity."""

    cache_identity: PersonaVisualCacheIdentity
    graph_identity: PersonaVisualIdentity | None
    asset_id: int
    asset_key: str
    asset_sha256: str
    manifest_frame_index: int
    selected_frame: int
    duration_ms: int | None
    width: int
    height: int
    paint_digest: str
    renderable: Pixels = field(repr=False, compare=False, hash=False)


def prepare_persona_buddy_frame(
    resolved_frame: PersonaVisualResolvedFrame,
    *,
    resolution_cache_identity: PersonaVisualCacheIdentity,
    cols: int,
    lines: int,
    max_cells: int | None = None,
) -> PersonaBuddyPreparedFrame:
    """Decode and scale the exact embedded frame into bounded Rich pixels.

    Raises:
        PersonaBuddyFrameError: If identity, decode, crop, or size validation fails.
    """

    try:
        if (
            type(resolved_frame) is not PersonaVisualResolvedFrame
            or type(resolution_cache_identity) is not PersonaVisualCacheIdentity
            or type(cols) is not int
            or type(lines) is not int
            or not 1 <= cols <= _MAX_COLS
            or not 1 <= lines <= _MAX_LINES
            or (max_cells is not None and (type(max_cells) is not int or max_cells < 1))
        ):
            raise ValueError
        data = resolved_frame.data
        if (
            type(data) is not bytes
            or not data
            or len(data) > _MAX_FRAME_BYTES
            or hashlib.sha256(data).hexdigest() != resolved_frame.sha256
        ):
            raise ValueError
        cache_asset = PersonaVisualCacheAsset(
            asset_id=resolved_frame.asset_id,
            asset_key=resolved_frame.asset_key,
            sha256=resolved_frame.sha256,
            manifest_frame_index=resolved_frame.manifest_frame_index,
            selected_frame=resolved_frame.selected_frame,
        )
        if cache_asset not in resolution_cache_identity.assets:
            raise ValueError

        with Image.open(BytesIO(data)) as source:
            if (
                source.width < 1
                or source.height < 1
                or source.width > MAX_ASSET_DIMENSION
                or source.height > MAX_ASSET_DIMENSION
                or type(resolved_frame.selected_frame) is not int
                or resolved_frame.selected_frame < 0
            ):
                raise ValueError
            source.seek(resolved_frame.selected_frame)
            source.load()
            image = source.convert("RGBA")

        region = resolved_frame.region
        if region is not None:
            if (
                region.x < 0
                or region.y < 0
                or region.width < 1
                or region.height < 1
                or region.x + region.width > image.width
                or region.y + region.height > image.height
            ):
                raise ValueError
            image = image.crop(
                (region.x, region.y, region.x + region.width, region.y + region.height)
            )

        image.thumbnail((cols, lines * 2), Image.Resampling.LANCZOS)
        if (
            image.width < 1
            or image.height < 1
            or (max_cells is not None and image.width * image.height > max_cells)
        ):
            raise ValueError
        return PersonaBuddyPreparedFrame(
            cache_identity=resolution_cache_identity,
            graph_identity=resolution_cache_identity.graph,
            asset_id=resolved_frame.asset_id,
            asset_key=resolved_frame.asset_key,
            asset_sha256=resolved_frame.sha256,
            manifest_frame_index=resolved_frame.manifest_frame_index,
            selected_frame=resolved_frame.selected_frame,
            duration_ms=resolved_frame.duration_ms,
            width=image.width,
            height=image.height,
            paint_digest=hashlib.sha256(image.tobytes()).hexdigest(),
            renderable=Pixels.from_image(image),
        )
    except PersonaBuddyFrameError:
        raise
    except Exception:
        raise PersonaBuddyFrameError() from None


def prepare_persona_buddy_portrait(
    portrait: PersonaVisualPortrait,
    *,
    resolution_cache_identity: PersonaVisualCacheIdentity,
    cols: int,
    lines: int,
) -> PersonaBuddyPreparedFrame:
    """Prepare the validated local Persona portrait fallback."""

    try:
        if (
            type(portrait) is not PersonaVisualPortrait
            or type(resolution_cache_identity) is not PersonaVisualCacheIdentity
            or resolution_cache_identity.portrait_id != portrait.portrait_id
            or resolution_cache_identity.portrait_revision != portrait.revision
            or resolution_cache_identity.portrait_sha256 != portrait.sha256
            or type(cols) is not int
            or type(lines) is not int
            or not 1 <= cols <= _MAX_COLS
            or not 1 <= lines <= _MAX_LINES
            or type(portrait.data) is not bytes
            or not portrait.data
            or len(portrait.data) > _MAX_FRAME_BYTES
            or hashlib.sha256(portrait.data).hexdigest() != portrait.sha256
            or portrait.selected_frame != 0
        ):
            raise ValueError
        with Image.open(BytesIO(portrait.data)) as source:
            if (
                source.width < 1
                or source.height < 1
                or source.width > MAX_ASSET_DIMENSION
                or source.height > MAX_ASSET_DIMENSION
            ):
                raise ValueError
            source.seek(0)
            source.load()
            image = source.convert("RGBA")
        image.thumbnail((cols, lines * 2), Image.Resampling.LANCZOS)
        return PersonaBuddyPreparedFrame(
            cache_identity=resolution_cache_identity,
            graph_identity=resolution_cache_identity.graph,
            asset_id=0,
            asset_key=portrait.portrait_id,
            asset_sha256=portrait.sha256,
            manifest_frame_index=-1,
            selected_frame=0,
            duration_ms=None,
            width=image.width,
            height=image.height,
            paint_digest=hashlib.sha256(image.tobytes()).hexdigest(),
            renderable=Pixels.from_image(image),
        )
    except PersonaBuddyFrameError:
        raise
    except Exception:
        raise PersonaBuddyFrameError() from None


__all__ = (
    "MAX_PERSONA_BUDDY_PREPARED_CELLS",
    "MAX_PERSONA_BUDDY_PREPARED_FRAMES",
    "PERSONA_BUDDY_FRAME_UNAVAILABLE",
    "PersonaBuddyFrameError",
    "PersonaBuddyPreparedFrame",
    "prepare_persona_buddy_frame",
    "prepare_persona_buddy_portrait",
)
