"""Characterize Media geometry through the shared adaptive-reader seam."""

from __future__ import annotations

import pytest

from tldw_chatbook.Utils.adaptive_reader_state import (
    LAYOUT_HYSTERESIS_WIDTH,
    PANE_GRIP_WIDTH,
    AdaptiveReaderEffectiveLayout,
    AdaptiveReaderLayoutPreferences,
    AdaptiveReaderLayoutProfile,
    normalize_adaptive_reader_preferences,
    resolve_adaptive_reader_layout,
)
from tldw_chatbook.Library.library_media_reader_state import (
    MEDIA_READER_LAYOUT_PROFILE,
    MediaReaderEffectiveLayout,
    MediaReaderLayoutPreferences,
    normalize_media_reader_preferences,
    resolve_media_reader_layout,
)
from tldw_chatbook.Library.library_rail_width import project_default_library_width


MEDIA_PROFILE = AdaptiveReaderLayoutProfile()


def test_media_compatibility_names_reexport_shared_layout_types() -> None:
    assert MediaReaderLayoutPreferences is AdaptiveReaderLayoutPreferences
    assert MediaReaderEffectiveLayout is AdaptiveReaderEffectiveLayout


def test_adaptive_profile_exposes_approved_widths() -> None:
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

    assert (
        normalize_adaptive_reader_preferences(raw)
        == (normalize_media_reader_preferences(raw))
        == AdaptiveReaderLayoutPreferences(
            library_open=False,
            items_open=True,
            custom_widths_enabled=True,
            library_width=48,
            items_width=32,
        )
    )


@pytest.mark.parametrize(
    ("width", "expected_geometry"),
    [
        (160, (True, True, 30, 40, 80)),
        (120, (True, True, 24, 40, 46)),
        (100, (False, True, 0, 46, 44)),
        (80, (False, False, 0, 0, 70)),
        (60, (False, False, 0, 0, 50)),
    ],
)
def test_shared_resolution_uses_adaptive_width_classes(
    width: int,
    expected_geometry: tuple[bool, bool, int, int, int],
) -> None:
    preferences = AdaptiveReaderLayoutPreferences()

    shared = resolve_adaptive_reader_layout(width, preferences, MEDIA_PROFILE)
    media = resolve_media_reader_layout(width, preferences)

    assert (
        shared.library_open,
        shared.items_open,
        shared.library_width,
        shared.items_width,
        shared.reader_width,
    ) == expected_geometry
    if width != 100:
        assert shared == media


def test_custom_width_above_comfort_is_not_shrunk_when_it_fits() -> None:
    preferences = normalize_adaptive_reader_preferences(
        {
            "custom_widths_enabled": True,
            "library_width": 36,
            "items_width": 64,
        }
    )

    shared = resolve_adaptive_reader_layout(130, preferences, MEDIA_PROFILE)

    assert (shared.library_width, shared.items_width, shared.reader_width) == (
        0,
        64,
        56,
    )


def test_comfort_growth_is_capped_by_profile_comfort_and_list_max() -> None:
    preferences = AdaptiveReaderLayoutPreferences(library_open=False)

    comfort_capped = resolve_adaptive_reader_layout(
        200,
        preferences,
        AdaptiveReaderLayoutProfile(list_comfort_width=56, list_max_width=72),
    )
    max_capped = resolve_adaptive_reader_layout(
        200,
        preferences,
        AdaptiveReaderLayoutProfile(list_comfort_width=80, list_max_width=60),
    )

    assert comfort_capped.items_width == 56
    assert max_capped.items_width == 60


def test_resolution_never_mutates_saved_preferences() -> None:
    preferences = AdaptiveReaderLayoutPreferences(
        library_open=False,
        items_open=True,
        custom_widths_enabled=True,
        library_width=36,
        items_width=40,
    )
    saved = preferences.__dict__.copy()

    layout = resolve_adaptive_reader_layout(120, preferences, MEDIA_PROFILE)

    assert layout.items_width == 56
    assert preferences.__dict__ == saved


def test_profile_work_minimum_is_protected_before_the_items_pane() -> None:
    editor_profile = AdaptiveReaderLayoutProfile(
        work_min_width=48,
        work_comfort_width=48,
    )

    layout = resolve_adaptive_reader_layout(
        97,
        AdaptiveReaderLayoutPreferences(library_open=False),
        editor_profile,
    )

    assert layout.items_open is False
    assert layout.reader_width == 87


def test_media_profile_protects_the_rendered_toolbar_work_minimum() -> None:
    layout = resolve_media_reader_layout(100, MediaReaderLayoutPreferences())

    assert MEDIA_READER_LAYOUT_PROFILE.work_min_width == 46
    assert layout.library_open is False
    assert layout.items_open is True
    assert layout.items_width == 44
    assert layout.reader_width >= 46
    assert layout.items_width + layout.reader_width == 90


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
    assert (
        shared.library_width,
        shared.items_width,
        shared.reader_width,
    ) == expected_widths
    assert shared.priority_pane == priority


@pytest.mark.parametrize("priority", ["library", "items"])
def test_explicit_open_priority_protects_the_requested_pane_when_possible(
    priority: str,
) -> None:
    layout = resolve_adaptive_reader_layout(
        120,
        AdaptiveReaderLayoutPreferences(),
        MEDIA_PROFILE,
        priority=priority,  # type: ignore[arg-type]
    )

    assert getattr(layout, f"{priority}_open") is True
    assert layout.priority_pane is None


def test_shared_resolution_preserves_hysteresis() -> None:
    preferences = AdaptiveReaderLayoutPreferences()
    collapsed = resolve_adaptive_reader_layout(117, preferences, MEDIA_PROFILE)

    boundary = resolve_adaptive_reader_layout(
        118,
        preferences,
        MEDIA_PROFILE,
        previous=collapsed,
    )
    reopened = resolve_adaptive_reader_layout(
        118 + LAYOUT_HYSTERESIS_WIDTH,
        preferences,
        MEDIA_PROFILE,
        previous=boundary,
    )

    assert boundary.library_open is False
    assert reopened.library_open is True


@pytest.mark.parametrize("width", [10, 11, 59, 60, 80, 100, 120, 122, 160])
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


def test_minimum_width_escape_keeps_work_mounted_without_changing_preferences() -> None:
    preferences = AdaptiveReaderLayoutPreferences()

    layout = resolve_adaptive_reader_layout(10, preferences, MEDIA_PROFILE)

    assert layout.library_open is False
    assert layout.items_open is False
    assert layout.reader_width == 0
    assert preferences == AdaptiveReaderLayoutPreferences()


def test_zero_width_is_a_pre_layout_sentinel_without_reading_previous_state() -> None:
    layout = resolve_adaptive_reader_layout(
        0,
        AdaptiveReaderLayoutPreferences(),
        MEDIA_PROFILE,
        previous=object(),  # type: ignore[arg-type]
    )

    assert layout == AdaptiveReaderEffectiveLayout(
        library_open=False,
        items_open=False,
        library_width=0,
        items_width=0,
        reader_width=0,
        priority_pane=None,
    )


@pytest.mark.parametrize("width", [116, 100, 80, 60])
def test_default_mode_projects_library_width_instead_of_using_dormant_saved_width(
    width: int,
) -> None:
    preferences = AdaptiveReaderLayoutPreferences(library_width=28)

    layout = resolve_adaptive_reader_layout(
        width,
        preferences,
        AdaptiveReaderLayoutProfile(work_min_width=48),
        priority="library",
    )

    requested_library_width = project_default_library_width(width)
    expected_library_width = (
        requested_library_width
        if width >= 2 * PANE_GRIP_WIDTH + requested_library_width + 48
        else min(24, max(width - 2 * PANE_GRIP_WIDTH, 0))
    )
    assert layout.library_width == expected_library_width


@pytest.mark.parametrize(
    ("width", "expected_items_width"),
    [(116, 56), (100, 42), (80, 32), (60, 32)],
)
def test_notes_navigator_explicit_items_priority_uses_projected_library_request(
    width: int, expected_items_width: int
) -> None:
    layout = resolve_adaptive_reader_layout(
        width,
        AdaptiveReaderLayoutPreferences(library_width=28),
        AdaptiveReaderLayoutProfile(work_min_width=48),
        priority="items",
    )

    assert layout.library_width == 0
    assert layout.items_width == expected_items_width
    assert layout.priority_pane == "items"


@pytest.mark.parametrize(
    ("width", "items_open"),
    [(116, True), (100, True), (80, False), (60, False)],
)
def test_notes_editor_preserves_work_before_items_at_production_widths(
    width: int, items_open: bool
) -> None:
    layout = resolve_adaptive_reader_layout(
        width,
        AdaptiveReaderLayoutPreferences(library_width=28),
        AdaptiveReaderLayoutProfile(work_min_width=48),
    )

    assert layout.items_open is items_open


@pytest.mark.parametrize(("width", "expected_library_width"), [(34, 24), (33, 23)])
def test_explicit_library_priority_keeps_both_grips_when_work_cannot_fit(
    width: int, expected_library_width: int
) -> None:
    layout = resolve_adaptive_reader_layout(
        width,
        AdaptiveReaderLayoutPreferences(library_open=False),
        AdaptiveReaderLayoutProfile(work_min_width=48),
        priority="library",
    )

    assert (layout.library_width, layout.items_width, layout.reader_width) == (
        expected_library_width,
        0,
        0,
    )


@pytest.mark.parametrize("custom_library_width", [24, 34, 35, 48])
@pytest.mark.parametrize(
    "profile",
    [
        AdaptiveReaderLayoutProfile(work_min_width=44),
        AdaptiveReaderLayoutProfile(work_min_width=46),
        AdaptiveReaderLayoutProfile(work_min_width=48),
    ],
    ids=["conversations", "media", "notes"],
)
def test_custom_mode_preserves_every_normalized_library_request_across_profiles(
    custom_library_width: int, profile: AdaptiveReaderLayoutProfile
) -> None:
    layout = resolve_adaptive_reader_layout(
        160,
        AdaptiveReaderLayoutPreferences(
            custom_widths_enabled=True,
            library_width=custom_library_width,
        ),
        profile,
    )

    assert layout.library_open is True
    assert layout.library_width == custom_library_width


@pytest.mark.parametrize(
    ("raw_width", "expected_width"),
    [
        (1, 24),
        (999, 48),
        ("not-a-number", 31),
        (True, 31),
        (None, 31),
    ],
)
def test_custom_width_normalization_uses_explicit_range_not_default_ceiling(
    raw_width: object, expected_width: int
) -> None:
    preferences = normalize_adaptive_reader_preferences(
        {"custom_widths_enabled": True, "library_width": raw_width}
    )

    assert preferences.library_width == expected_width
