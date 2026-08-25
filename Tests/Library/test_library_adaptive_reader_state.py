"""Characterize Media geometry through the shared adaptive-reader seam."""

from __future__ import annotations

import pytest

from tldw_chatbook.Library.library_adaptive_reader_state import (
    LAYOUT_HYSTERESIS_WIDTH,
    PANE_GRIP_WIDTH,
    AdaptiveReaderEffectiveLayout,
    AdaptiveReaderLayoutPreferences,
    AdaptiveReaderLayoutProfile,
    normalize_adaptive_reader_preferences,
    resolve_adaptive_reader_layout,
)
from tldw_chatbook.Library.library_media_reader_state import (
    MediaReaderEffectiveLayout,
    MediaReaderLayoutPreferences,
    normalize_media_reader_preferences,
    resolve_media_reader_layout,
)


MEDIA_PROFILE = AdaptiveReaderLayoutProfile()


def test_media_compatibility_names_reexport_shared_layout_types() -> None:
    assert MediaReaderLayoutPreferences is AdaptiveReaderLayoutPreferences
    assert MediaReaderEffectiveLayout is AdaptiveReaderEffectiveLayout


def test_adaptive_profile_exposes_approved_future_widths_without_expansion() -> None:
    assert MEDIA_PROFILE == AdaptiveReaderLayoutProfile(
        list_min_width=32,
        list_target_width=40,
        list_comfort_width=56,
        list_max_width=72,
        work_min_width=44,
        work_comfort_width=44,
    )


def test_shared_normalization_matches_current_media_custom_width_behavior() -> None:
    raw = {
        "library_open": "false",
        "items_open": "true",
        "custom_widths_enabled": True,
        "library_width": 999,
        "items_width": 1,
    }

    assert normalize_adaptive_reader_preferences(raw) == (
        normalize_media_reader_preferences(raw)
    ) == AdaptiveReaderLayoutPreferences(
        library_open=False,
        items_open=True,
        custom_widths_enabled=True,
        library_width=48,
        items_width=32,
    )


@pytest.mark.parametrize(
    ("width", "expected_geometry"),
    [
        (160, (True, True, 28, 40, 82)),
        (120, (False, True, 0, 40, 70)),
        (80, (False, False, 0, 0, 70)),
        (60, (False, False, 0, 0, 50)),
    ],
)
def test_shared_resolution_matches_media_at_current_width_classes(
    width: int,
    expected_geometry: tuple[bool, bool, int, int, int],
) -> None:
    preferences = AdaptiveReaderLayoutPreferences()

    shared = resolve_adaptive_reader_layout(width, preferences, MEDIA_PROFILE)
    media = resolve_media_reader_layout(width, preferences)

    assert shared == media
    assert (
        shared.library_open,
        shared.items_open,
        shared.library_width,
        shared.items_width,
        shared.reader_width,
    ) == expected_geometry


def test_shared_resolution_preserves_current_custom_width_geometry() -> None:
    preferences = normalize_adaptive_reader_preferences(
        {
            "custom_widths_enabled": True,
            "library_width": 36,
            "items_width": 60,
        }
    )

    shared = resolve_adaptive_reader_layout(140, preferences, MEDIA_PROFILE)
    media = resolve_media_reader_layout(140, preferences)

    assert shared == media
    assert (shared.library_width, shared.items_width, shared.reader_width) == (
        0,
        60,
        70,
    )


@pytest.mark.parametrize(
    ("priority", "expected_widths"),
    [("library", (24, 0, 26)), ("items", (0, 32, 18))],
)
def test_shared_resolution_preserves_explicit_collapse_priority(
    priority: str,
    expected_widths: tuple[int, int, int],
) -> None:
    preferences = AdaptiveReaderLayoutPreferences()

    shared = resolve_adaptive_reader_layout(
        60,
        preferences,
        MEDIA_PROFILE,
        priority=priority,  # type: ignore[arg-type]
    )
    media = resolve_media_reader_layout(
        60,
        preferences,
        priority=priority,  # type: ignore[arg-type]
    )

    assert shared == media
    assert (
        shared.library_width,
        shared.items_width,
        shared.reader_width,
    ) == expected_widths
    assert shared.priority_pane == priority


def test_shared_resolution_preserves_current_hysteresis() -> None:
    preferences = AdaptiveReaderLayoutPreferences()
    collapsed = resolve_adaptive_reader_layout(121, preferences, MEDIA_PROFILE)

    boundary = resolve_adaptive_reader_layout(
        122,
        preferences,
        MEDIA_PROFILE,
        previous=collapsed,
    )
    reopened = resolve_adaptive_reader_layout(
        122 + LAYOUT_HYSTERESIS_WIDTH,
        preferences,
        MEDIA_PROFILE,
        previous=boundary,
    )

    assert boundary == resolve_media_reader_layout(
        122,
        preferences,
        previous=collapsed,
    )
    assert boundary.library_open is False
    assert reopened.library_open is True


@pytest.mark.parametrize("width", [10, 11, 59, 60, 80, 120, 122, 160])
def test_shared_geometry_is_non_negative_and_stays_within_width_budget(
    width: int,
) -> None:
    layout = resolve_adaptive_reader_layout(
        width,
        AdaptiveReaderLayoutPreferences(),
        MEDIA_PROFILE,
    )

    assert min(layout.library_width, layout.items_width, layout.reader_width) >= 0
    assert (
        layout.library_width
        + layout.items_width
        + layout.reader_width
        + 2 * PANE_GRIP_WIDTH
        <= width
    )
