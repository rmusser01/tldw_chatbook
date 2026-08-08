"""Preview-eligibility policy caps (task-3401.9)."""

from tldw_chatbook.Media_Playback.preview_policy import (
    PREVIEW_MAX_DURATION_SECONDS,
    PREVIEW_MAX_HEIGHT,
    PREVIEW_MAX_WIDTH,
    check_preview_eligibility,
)


def test_within_caps_eligible():
    result = check_preview_eligibility(duration_seconds=6.0, width=1920, height=1080)
    assert result.eligible and result.reason == ""


def test_unknown_shape_passes():
    # None values never block here -- the frame source re-checks its own
    # probed values at open time.
    assert check_preview_eligibility(
        duration_seconds=None, width=None, height=None
    ).eligible


def test_duration_cap_enforced():
    result = check_preview_eligibility(
        duration_seconds=PREVIEW_MAX_DURATION_SECONDS + 1, width=None, height=None
    )
    assert not result.eligible
    assert "previews cap" in result.reason
    edge = check_preview_eligibility(
        duration_seconds=PREVIEW_MAX_DURATION_SECONDS, width=None, height=None
    )
    assert edge.eligible


def test_width_cap_enforced():
    result = check_preview_eligibility(
        duration_seconds=5, width=PREVIEW_MAX_WIDTH + 2, height=1080
    )
    assert not result.eligible and "wide" in result.reason


def test_height_cap_enforced():
    result = check_preview_eligibility(
        duration_seconds=5, width=1920, height=PREVIEW_MAX_HEIGHT + 2
    )
    assert not result.eligible and "tall" in result.reason
