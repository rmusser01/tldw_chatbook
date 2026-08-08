"""Pure preview-eligibility policy for in-transcript video previews (task-3401.9).

No I/O, no av import -- the caps a preview must satisfy BEFORE any decode
work happens, so an oversized clip is refused without burning a frame of
CPU. Previews are deliberately SHORT and SMALL (ADR-044 §5A): the in-app
surface is a glanceable preview, not the player (that is task-3401.10's
full player screen, which has no such caps).
"""

from __future__ import annotations

from dataclasses import dataclass

#: Hard caps for in-transcript previews.
PREVIEW_MAX_DURATION_SECONDS = 30.0
PREVIEW_MAX_WIDTH = 2560
PREVIEW_MAX_HEIGHT = 1440
PREVIEW_TARGET_FPS = 12.0


@dataclass(frozen=True)
class PreviewEligibility:
    """Whether a clip may be previewed in-transcript, and why not.

    Attributes:
        eligible: True when every cap passes.
        reason: Human-readable refusal reason when not eligible (rendered
            in the card instead of the preview); empty when eligible.
    """

    eligible: bool
    reason: str = ""


def check_preview_eligibility(
    *,
    duration_seconds: float | None,
    width: int | None,
    height: int | None,
) -> PreviewEligibility:
    """Evaluate the preview caps against a clip's probed/recorded shape.

    Unknown values (``None``) pass -- they never widen what is decoded
    (the frame source re-checks its own probed values at open time and
    refuses there too), so a metadata-light clip can still be previewed
    when its real shape turns out within caps.

    Args:
        duration_seconds: Clip length, or ``None`` when unknown.
        width: Frame width in pixels, or ``None``.
        height: Frame height in pixels, or ``None``.

    Returns:
        A :class:`PreviewEligibility`; ``reason`` names the first failed cap.
    """
    if duration_seconds is not None and duration_seconds > PREVIEW_MAX_DURATION_SECONDS:
        return PreviewEligibility(
            False,
            f"clip is {duration_seconds:g}s -- previews cap at "
            f"{PREVIEW_MAX_DURATION_SECONDS:g}s (play it with the system player)",
        )
    if width is not None and width > PREVIEW_MAX_WIDTH:
        return PreviewEligibility(
            False,
            f"frame is {width}px wide -- previews cap at {PREVIEW_MAX_WIDTH}px "
            "(play it with the system player)",
        )
    if height is not None and height > PREVIEW_MAX_HEIGHT:
        return PreviewEligibility(
            False,
            f"frame is {height}px tall -- previews cap at {PREVIEW_MAX_HEIGHT}px "
            "(play it with the system player)",
        )
    return PreviewEligibility(True)
