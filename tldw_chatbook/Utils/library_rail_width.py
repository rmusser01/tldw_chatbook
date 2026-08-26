"""Pure bounded-fractional width policy for the Library rail.

This config-safe leaf intentionally lives in ``Utils`` beside adaptive reader
state so importing application configuration does not initialize the Library
feature package.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

LIBRARY_REFERENCE_WIDTH = 31
LIBRARY_MIN_WIDTH = 24
LIBRARY_DEFAULT_MAX_WIDTH = 34
LIBRARY_CUSTOM_MAX_WIDTH = 48
LIBRARY_CANVAS_MIN_WIDTH = 40
LIBRARY_EMERGENCY_WIDTH = LIBRARY_MIN_WIDTH + LIBRARY_CANVAS_MIN_WIDTH


class OrdinaryRailPresentation(Enum):
    """The ordinary Library rail's effective presentation."""

    ALONGSIDE = "alongside"
    RAIL_ONLY = "rail_only"
    HIDDEN = "hidden"


@dataclass(frozen=True)
class OrdinaryRailStyleContract:
    """Inline style declarations for one ordinary Library rail presentation."""

    display: bool
    width: str | int | None
    min_width: int | None
    max_width: int | None


def _require_positive_content_width(content_width: int) -> None:
    if type(content_width) is not int or content_width < 1:
        raise ValueError("content_width must be a positive integer.")


def project_default_library_width(content_width: int) -> int:
    """Project the bounded 3:13 default rail width from available content.

    Args:
        content_width: Positive available Library shell width in terminal cells.

    Returns:
        The 3:13 projection clamped to the default 24–34 cell range.

    Raises:
        ValueError: If ``content_width`` is not a positive integer, including bool.
    """
    _require_positive_content_width(content_width)
    fractional_width = (3 * content_width + 8) // 16
    return min(max(fractional_width, LIBRARY_MIN_WIDTH), LIBRARY_DEFAULT_MAX_WIDTH)


def ordinary_emergency_required(content_width: int) -> bool:
    """Return whether ordinary Library content must use an emergency takeover.

    Args:
        content_width: Positive available Library shell width in terminal cells.

    Returns:
        Whether the width is below the rail-and-canvas minimum of 64 cells.

    Raises:
        ValueError: If ``content_width`` is not a positive integer, including bool.
    """
    _require_positive_content_width(content_width)
    return content_width < LIBRARY_EMERGENCY_WIDTH


def _validate_saved_width(saved_width: int) -> None:
    if type(saved_width) is not int:
        raise TypeError("saved_width must be a normalized integer.")
    if not LIBRARY_MIN_WIDTH <= saved_width <= LIBRARY_CUSTOM_MAX_WIDTH:
        raise ValueError(
            "saved_width must be between "
            f"{LIBRARY_MIN_WIDTH} and {LIBRARY_CUSTOM_MAX_WIDTH}."
        )


def resolve_ordinary_rail_contract(
    content_width: int,
    presentation: OrdinaryRailPresentation,
    custom_widths_enabled: bool,
    saved_width: int,
) -> OrdinaryRailStyleContract:
    """Resolve pure inline style declarations for the ordinary Library rail.

    Args:
        content_width: Positive available Library shell width in terminal cells.
        presentation: Effective alongside, rail-only, or hidden presentation.
        custom_widths_enabled: Exact boolean controlling saved-width use.
        saved_width: Normalized persisted width from 24 through 48 inclusive.

    Returns:
        Immutable declarations; ``None`` clears a future Textual inline style rule.

    Raises:
        TypeError: If presentation, the custom flag, or saved width has an invalid
            type; bool is not accepted as a saved width.
        ValueError: If a width is invalid or alongside cannot fit safely.
    """
    _require_positive_content_width(content_width)
    if not isinstance(presentation, OrdinaryRailPresentation):
        raise TypeError("presentation must be OrdinaryRailPresentation.")
    if type(custom_widths_enabled) is not bool:
        raise TypeError("custom_widths_enabled must be a boolean.")
    _validate_saved_width(saved_width)

    if presentation is OrdinaryRailPresentation.RAIL_ONLY:
        return OrdinaryRailStyleContract(True, "1fr", 0, None)
    if presentation is OrdinaryRailPresentation.HIDDEN:
        return OrdinaryRailStyleContract(False, None, None, None)
    if ordinary_emergency_required(content_width):
        raise ValueError(
            "alongside presentation requires content_width of at least 64."
        )
    if not custom_widths_enabled:
        return OrdinaryRailStyleContract(
            True, "3fr", LIBRARY_MIN_WIDTH, LIBRARY_DEFAULT_MAX_WIDTH
        )

    effective_width = max(
        LIBRARY_MIN_WIDTH,
        min(saved_width, content_width - LIBRARY_CANVAS_MIN_WIDTH),
    )
    return OrdinaryRailStyleContract(
        True, effective_width, effective_width, effective_width
    )
