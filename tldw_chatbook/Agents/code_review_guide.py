"""Complete packaged primary recipes with bounded canonical envelopes (ADR-281)."""

from importlib.resources import files
from typing import Literal

from .code_review_models import (
    GUIDE_MAX_BYTES,
    ReviewRefused,
    ReviewTeam,
    canonical_record_bytes,
    review_result_char_limit,
)


def fit_code_review_guide(result: dict, max_result_chars: int) -> dict:
    """Admit the entire envelope or return a body-free bounded diagnostic."""
    limit = review_result_char_limit(max_result_chars)
    encoded = canonical_record_bytes(result)
    if len(encoded) <= GUIDE_MAX_BYTES and len(encoded.decode("utf-8")) <= limit:
        return result
    diagnostic = {
        "schema_version": 1,
        "diagnostic": "review_guide_capacity_unavailable",
    }
    if len(canonical_record_bytes(diagnostic).decode("utf-8")) > limit:
        raise ReviewRefused("review_response_capacity_unavailable")
    return diagnostic


def load_code_review_guide(
    stage: Literal["review", "verify"],
    team: ReviewTeam,
    *,
    max_result_chars: int = 16000,
) -> dict:
    """Read one complete fixed recipe, including safe team setup diagnostics.

    Args:
        stage: Closed primary recipe stage.
        team: Saved stable preset references, never provider credentials.
        max_result_chars: Trusted live runtime result limit.

    Returns:
        A versioned envelope within 12 KiB and the runtime character cap, or
        an explicit diagnostic without truncated instructions.

    Raises:
        ReviewRefused: For invalid stage, unavailable resources or insufficient
            space even for the diagnostic.
    """
    if stage not in ("review", "verify"):
        raise ReviewRefused("invalid_stage")
    try:
        resource = files("tldw_chatbook.Agents").joinpath(
            f"code_review_guides/{stage}.md"
        )
        with resource.open("rb") as handle:
            raw = handle.read(GUIDE_MAX_BYTES + 1)
        if len(raw) > GUIDE_MAX_BYTES:
            return fit_code_review_guide(
                {"text": "x" * (GUIDE_MAX_BYTES + 1)}, max_result_chars
            )
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise ReviewRefused("review_guide_unavailable") from exc
    if not text.strip():
        raise ReviewRefused("review_guide_unavailable")
    result = {
        "schema_version": 1,
        "stage": stage,
        "text": text,
        "team": team.model_dump(mode="json"),
        "diagnostics": ["review_team_setup_required"]
        if len(team.reviewers) < 2
        else [],
    }
    return fit_code_review_guide(result, max_result_chars)
