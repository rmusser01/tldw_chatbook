"""Existing Personas thumbnail helpers shared by the screen's public seams."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.css.query import QueryError

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
