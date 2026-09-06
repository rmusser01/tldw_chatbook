"""Characterize Media geometry through the shared adaptive-reader seam."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tldw_chatbook.Utils.adaptive_reader_state import (
    LAYOUT_HYSTERESIS_WIDTH,
    PANE_GRIP_WIDTH,
    READER_COMFORT_WIDTH,
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
from tldw_chatbook.UI.Library_Modules import screen_constants
from tldw_chatbook.UI.Library_Modules.screen_constants import (
    LIBRARY_COLLECTIONS_READER_PROFILE,
    LIBRARY_CONVERSATION_READER_PROFILE,
    LIBRARY_SKILLS_READER_PROFILE,
)
from tldw_chatbook.Utils.library_rail_width import project_default_library_width


MEDIA_PROFILE = AdaptiveReaderLayoutProfile()

# The control for every "did list growth change this?" comparison: Media's own
# profile with the one knob under test turned off, so a second divergence
# (task-31633 AC#2 narrowed Media's grips to one cell) can never be read as
# list growth.
MEDIA_PROFILE_WITHOUT_GROWTH = replace(MEDIA_READER_LAYOUT_PROFILE, list_grows=False)


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
        list_grows=False,
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
    ("width", "expected_geometry", "expected_media_geometry"),
    [
        # task-31633: the Media profile shares the Reader's surplus with its
        # Items column, so it diverges from the generic profile wherever the
        # Reader sits above its 46-cell minimum -- and AC#2 narrowed Media's
        # two grips from five cells each to one, so every Media row below also
        # carries the eight cells they gave back. The generic column is
        # untouched by both.
        (160, (True, True, 30, 40, 80), (True, True, 30, 56, 72)),
        (120, (True, True, 24, 40, 46), (True, True, 24, 44, 50)),
        (100, (False, True, 0, 46, 44), (False, True, 0, 52, 46)),
        (80, (False, False, 0, 0, 70), (False, False, 0, 0, 78)),
        (60, (False, False, 0, 0, 50), (False, False, 0, 0, 58)),
    ],
)
def test_shared_resolution_uses_adaptive_width_classes(
    width: int,
    expected_geometry: tuple[bool, bool, int, int, int],
    expected_media_geometry: tuple[bool, bool, int, int, int],
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
    assert (
        media.library_open,
        media.items_open,
        media.library_width,
        media.items_width,
        media.reader_width,
    ) == expected_media_geometry


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
    # 44 / 90 while Media's two grips still cost five cells each; the eight
    # they gave back land on the list, because the Reader is on its minimum.
    assert layout.items_width == 52
    assert layout.reader_width >= 46
    assert layout.items_width + layout.reader_width == 98


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


# ---------------------------------------------------------------------------
# task-31633 AC#1/AC#4: the Media Items column grows with the terminal once the
# Reader is comfortable.
#
# Critique #5 P1 measured the inversion: at 235x52 the Items list was 40 cells
# and truncated a 98-character title after 31, while at 100x30 the same list
# was 44 cells and truncated after 39 -- the wider terminal got the narrower
# list, because every cell past the two panes' nominal widths went to the
# Reader. `resolve_adaptive_reader_layout` is shared by four destinations, so
# growth is opt-in per profile and the sibling surfaces are pinned below at the
# two review widths and at their own library-open edge.
# ---------------------------------------------------------------------------

# The three pinned below are the ones with explicit geometry tuples; the
# opt-in guard sweeps EVERY profile constant the Library screen declares
# (Notes, File Notes and Prompts included) so a future destination cannot
# quietly inherit growth.
SIBLING_PROFILES = {
    "conversations": LIBRARY_CONVERSATION_READER_PROFILE,
    "skills": LIBRARY_SKILLS_READER_PROFILE,
    "collections": LIBRARY_COLLECTIONS_READER_PROFILE,
}
DECLARED_PROFILES = {
    name: value
    for name, value in vars(screen_constants).items()
    if isinstance(value, AdaptiveReaderLayoutProfile)
}


def _pane_widths(
    layout: AdaptiveReaderEffectiveLayout,
) -> tuple[bool, bool, int, int, int]:
    return (
        layout.library_open,
        layout.items_open,
        layout.library_width,
        layout.items_width,
        layout.reader_width,
    )


def test_only_the_media_profile_opts_into_list_growth() -> None:
    assert MEDIA_READER_LAYOUT_PROFILE.list_grows is True
    # task-31633 AC#2: the one-cell grip is opt-in the same way.
    assert MEDIA_READER_LAYOUT_PROFILE.grip_width == 1
    for name, profile in DECLARED_PROFILES.items():
        assert profile.grip_width == PANE_GRIP_WIDTH, name
    assert AdaptiveReaderLayoutProfile().grip_width == PANE_GRIP_WIDTH
    assert set(SIBLING_PROFILES.values()) <= set(DECLARED_PROFILES.values())
    assert len(DECLARED_PROFILES) >= 6, sorted(DECLARED_PROFILES)
    for name, profile in DECLARED_PROFILES.items():
        assert profile.list_grows is False, name
    assert AdaptiveReaderLayoutProfile().list_grows is False


@pytest.mark.parametrize(
    ("surface", "width", "expected"),
    [
        # Recorded from the resolver at badff73f1, before list growth existed.
        ("conversations", 100, (False, True, 0, 46, 44)),
        ("conversations", 117, (False, True, 0, 56, 51)),
        ("conversations", 118, (True, True, 24, 40, 44)),
        ("conversations", 235, (True, True, 34, 40, 151)),
        ("skills", 100, (False, True, 0, 42, 48)),
        ("skills", 121, (False, True, 0, 56, 55)),
        ("skills", 122, (True, True, 24, 40, 48)),
        ("skills", 235, (True, True, 34, 40, 151)),
        ("collections", 100, (False, True, 0, 42, 48)),
        ("collections", 121, (False, True, 0, 56, 55)),
        ("collections", 122, (True, True, 24, 40, 48)),
        ("collections", 235, (True, True, 34, 40, 151)),
    ],
)
def test_sibling_reader_layouts_are_untouched_by_media_list_growth(
    surface: str, width: int, expected: tuple[bool, bool, int, int, int]
) -> None:
    layout = resolve_adaptive_reader_layout(
        width,
        AdaptiveReaderLayoutPreferences(),
        SIBLING_PROFILES[surface],
    )

    assert _pane_widths(layout) == expected


@pytest.mark.parametrize(
    ("width", "expected"),
    [
        # The Reader sits on its 46-cell minimum at both of these, so list
        # growth has no surplus to share: every cell of movement from
        # (0, 44, 46) / (24, 40, 46) is the eight the one-cell grips gave back
        # (task-31633 AC#2), which is why the list -- not the Reader -- has
        # them.
        (100, (False, True, 0, 52, 46)),
        (120, (True, True, 24, 44, 50)),
        # 119 used to close the rail and put the list on its 56-cell comfort
        # ceiling: (False, True, 0, 56, 53). Eight more cells is enough for the
        # rail to stay open there, so this is the one width where the grips
        # changed which panes are open, not just how wide they are.
        (119, (True, True, 24, 43, 50)),
    ],
)
def test_media_layout_where_list_growth_has_no_surplus_to_share(
    width: int, expected: tuple[bool, bool, int, int, int]
) -> None:
    layout = resolve_media_reader_layout(width, MediaReaderLayoutPreferences())

    assert _pane_widths(layout) == expected


@pytest.mark.parametrize(
    ("width", "expected"),
    [
        # 235: was (True, True, 34, 40, 151) -- 105 surplus cells all went to
        # the Reader. 122 is the narrowest width where growth moves a cell:
        # was (True, True, 24, 40, 48). Both rows then gained the eight cells
        # the one-cell grips gave back (task-31633 AC#2): 135 -> 143 on the
        # Reader at 235, and 41 -> 45 on the list at 122.
        (235, (True, True, 34, 56, 143)),
        (122, (True, True, 24, 45, 51)),
    ],
)
def test_media_items_column_grows_once_the_reader_is_comfortable(
    width: int, expected: tuple[bool, bool, int, int, int]
) -> None:
    layout = resolve_media_reader_layout(width, MediaReaderLayoutPreferences())

    assert _pane_widths(layout) == expected


@pytest.mark.parametrize("custom_items_width", [32, 34, 48])
@pytest.mark.parametrize("width", [160, 235])
def test_a_typed_custom_items_width_is_obeyed_rather_than_grown(
    width: int, custom_items_width: int
) -> None:
    """Settings > Appearance > Custom widths is a hand-typed number.

    "Automatic" adapts; "Custom" obeys. Without the gate the typed value was
    silently overridden above ~130 columns (review Important 1).
    """
    custom = MediaReaderLayoutPreferences(
        custom_widths_enabled=True,
        library_width=31,
        items_width=custom_items_width,
    )

    grown = resolve_media_reader_layout(width, custom)
    ungrown = resolve_adaptive_reader_layout(width, custom, MEDIA_PROFILE_WITHOUT_GROWTH)

    assert grown.items_width == custom_items_width
    assert grown == ungrown


def test_media_items_column_is_wider_at_235_than_at_100() -> None:
    narrow = resolve_media_reader_layout(100, MediaReaderLayoutPreferences())
    wide = resolve_media_reader_layout(235, MediaReaderLayoutPreferences())

    assert wide.items_width >= 47
    assert wide.items_width > narrow.items_width
    assert wide.reader_width >= READER_COMFORT_WIDTH


@pytest.mark.parametrize("width", range(60, 301))
def test_list_growth_never_shrinks_the_list_or_starves_the_reader(
    width: int,
) -> None:
    preferences = MediaReaderLayoutPreferences()
    ungrown = resolve_adaptive_reader_layout(
        width, preferences, MEDIA_PROFILE_WITHOUT_GROWTH
    )
    grown = resolve_media_reader_layout(width, preferences)

    assert grown.items_width >= ungrown.items_width
    assert grown.items_width <= max(
        min(
            MEDIA_READER_LAYOUT_PROFILE.list_comfort_width,
            MEDIA_READER_LAYOUT_PROFILE.list_max_width,
        ),
        ungrown.items_width,
    )
    assert (grown.library_open, grown.items_open) == (
        ungrown.library_open,
        ungrown.items_open,
    )
    assert grown.library_width == ungrown.library_width
    if grown.items_open:
        assert grown.reader_width >= READER_COMFORT_WIDTH
        assert grown.reader_width >= MEDIA_READER_LAYOUT_PROFILE.work_min_width
    assert (
        grown.library_width
        + grown.items_width
        + grown.reader_width
        + 2 * MEDIA_READER_LAYOUT_PROFILE.grip_width
    ) == width
