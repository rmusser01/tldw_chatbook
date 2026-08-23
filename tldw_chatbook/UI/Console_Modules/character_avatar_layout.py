"""Pure cell geometry for the Console Character avatar."""

from __future__ import annotations

from typing import Any

from ...Chat.console_image_view import (
    fit_image_cell_size,
    scale_image_for_cell_box,
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

    scaled = scale_image_for_cell_box(image, box_cols, box_lines)
    scaled_width = max(1, int(scaled.width))
    scaled_height = max(1, int(scaled.height))
    intrinsic_cols = min(box_cols, scaled_width)
    intrinsic_lines = min(box_lines, max(1, (scaled_height + 1) // 2))
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


__all__ = ["fit_character_avatar_cell_box"]
