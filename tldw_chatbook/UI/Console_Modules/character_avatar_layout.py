"""Pure cell geometry and off-loop pixel work for the Console Character avatar."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

from ...Chat.console_image_view import (
    fit_image_cell_size,
    scale_image_pixel_size_for_cell_box,
)
from ...Utils.mosaic_render import mosaic_contain_cell_size


def fit_character_avatar_cell_box(
    image: Any,
    available_cols: int,
    available_lines: int,
) -> tuple[int, int]:
    """Return a scale-down-only contain box for ``image``.

    The shared image scaler's ``thumbnail`` operation never enlarges its
    source. Its dimensions cap the intrinsic terminal footprint; the shared
    cell fitter still receives the original dimensions so thumbnail rounding
    cannot change the source aspect ratio by a row or column. The shared mosaic
    contain-grid helper canonicalizes the result to the exact fallback grid;
    graphics then uses that same box.

    TASK-22221: those capping dimensions come from
    ``scale_image_pixel_size_for_cell_box`` -- arithmetic identical to what
    ``thumbnail`` would produce -- rather than from an actual LANCZOS resample
    whose pixels were discarded. The rail runs this once per distinct viewport
    size during a resize drag, on the event loop.

    Args:
        image: Decoded PIL-compatible image. The source is never modified.
        available_cols: Measured Character-body columns available to the image.
        available_lines: Rows left beneath the 35-row complete-body ceiling.

    Returns:
        ``(width_cells, height_cells)``. ``(0, 0)`` means no image cell fits.
    """

    box_cols = max(0, int(available_cols))
    box_lines = max(0, int(available_lines))
    if box_cols == 0 or box_lines == 0:
        return 0, 0

    scaled_width, scaled_height = scale_image_pixel_size_for_cell_box(
        image.width,
        image.height,
        box_cols,
        box_lines,
    )
    intrinsic_cols = min(box_cols, max(1, int(scaled_width)))
    intrinsic_lines = min(box_lines, max(1, (max(1, int(scaled_height)) + 1) // 2))
    fitted_cols, fitted_lines = fit_image_cell_size(
        max(1, int(image.width)),
        max(1, int(image.height)),
        intrinsic_cols,
        intrinsic_lines,
    )
    return mosaic_contain_cell_size(
        image.width,
        image.height,
        fitted_cols,
        fitted_lines,
    )


@dataclass(frozen=True)
class CharacterAvatarPrerender:
    """One avatar renderable built off the event loop, plus its identity.

    The consumer re-derives the identity on the loop and uses ``renderable``
    only on an exact match, so a render that completed against a stale image,
    box, or colour mode degrades to an inline rebuild instead of painting the
    wrong thing (TASK-22221's completion race).
    """

    image: Any
    box: tuple[int, int]
    monochrome: bool
    renderable: Any

    def matches(self, image: Any, box: tuple[int, int], monochrome: bool) -> bool:
        """Whether this render is still exactly the one the caller needs.

        Args:
            image: The image the caller is about to render.
            box: The caller's resolved ``(cols, lines)`` cell box.
            monochrome: The caller's current colour mode.

        Returns:
            True only when this render was built from the same image object,
            for the same box, in the same colour mode.
        """

        return (
            self.image is image
            and self.box == (int(box[0]), int(box[1]))
            and self.monochrome is bool(monochrome)
        )


def render_character_avatar_mosaic(
    image: Any,
    box_cols: int,
    box_lines: int,
    *,
    monochrome: bool = False,
) -> Any:
    """Bake the rail avatar's non-graphics renderable from a PIL image.

    Quadrant mosaic (2x2 subpixels per cell) at the rail's fitted box --
    double the horizontal detail of a half-block Pixels build with the same
    universal Block Elements font coverage. Pure: no Textual, no DOM, no
    shared state, so it is safe to call from a worker thread.

    Args:
        image: The decoded portrait; never modified.
        box_cols: Target width in columns (task-1661: rail-derived).
        box_lines: Target height in lines.
        monochrome: Carry the image in shade GLYPHS rather than background
            colour. The coloured mosaic is spaces styled ``on rgb(...)``, so
            with colour unavailable it renders as a blank box -- the avatar
            does not degrade, it disappears. Textual switches the whole app
            to monochrome when ``NO_COLOR`` is set, which is one confirmed
            way a user sees no portrait at all.

    Returns:
        A Rich renderable for the fitted box.
    """

    # Imported inside the call so tests can patch the shared renderer at its
    # own module (the recovery-path tests do exactly that).
    from ...Utils.mosaic_render import mosaic_from_image

    return mosaic_from_image(
        image, box_cols, box_lines, fit="contain", monochrome=monochrome
    )


def prerender_character_avatar(
    image: Any,
    box: tuple[int, int],
    monochrome: bool,
) -> CharacterAvatarPrerender | None:
    """Resolve the fitted box and render its mosaic, off the event loop.

    Runs the whole pixel leg a viewport change would otherwise pay on the
    loop (measured 6.29 ms median for a 1024px card, 187.2 ms across a
    37-step resize drag). Applies ``fit_character_avatar_cell_box`` exactly
    once to ``box`` -- the same single transformation the widget builder
    applies -- so both sides resolve the identical final box.

    Args:
        image: The decoded portrait; never modified.
        box: The rail's target ``(cols, lines)`` cell box.
        monochrome: Whether the app is currently rendering without colour.

    Returns:
        A prerender token, or ``None`` when no image cell fits.
    """

    resolved = fit_character_avatar_cell_box(image, box[0], box[1])
    if resolved == (0, 0):
        return None
    return CharacterAvatarPrerender(
        image=image,
        box=resolved,
        monochrome=bool(monochrome),
        renderable=render_character_avatar_mosaic(
            image, resolved[0], resolved[1], monochrome=bool(monochrome)
        ),
    )


def build_character_avatar_prerender_job(
    spec: Any,
    box: tuple[int, int] | None,
    *,
    monochrome: bool,
) -> Callable[[], CharacterAvatarPrerender | None] | None:
    """Return a thread-safe render job for ``spec`` at ``box``, if any.

    Called ON THE EVENT LOOP so the live spec and colour mode are read there;
    the returned zero-arg callable closes over immutable values only and is
    what the rail hands to a worker thread (TASK-22221).

    Args:
        spec: The screen's current avatar spec, or None.
        box: The rail's target cell box, or None.
        monochrome: Whether the app is currently rendering without colour.

    Returns:
        A zero-arg job, or ``None`` when this spec/box has no pixel leg worth
        moving off the loop -- no image, an already-baked renderable, a
        graphics-mode portrait (whose scaling belongs to that renderer), or
        an empty box.
    """

    spec = spec or {}
    image = spec.get("pil")
    if (
        image is None
        or spec.get("pixels") is not None
        or spec.get("mode") == "graphics"
        or not box
        or tuple(box) == (0, 0)
    ):
        return None
    return partial(
        # Resolved through the module global at job-build time so the
        # renderer stays one patchable seam.
        globals()["prerender_character_avatar"],
        image,
        (int(box[0]), int(box[1])),
        bool(monochrome),
    )


__all__ = [
    "CharacterAvatarPrerender",
    "build_character_avatar_prerender_job",
    "fit_character_avatar_cell_box",
    "prerender_character_avatar",
    "render_character_avatar_mosaic",
]
