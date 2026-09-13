"""Characterize Media geometry through the shared adaptive-reader seam."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tldw_chatbook.Utils.adaptive_reader_state import (
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
        list_target_width=50,
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
        # Five-cell controls remain reserved; extra default pane width yields
        # before a pane collapses. Media still shares reader surplus.
        (160, (True, True, 35, 50, 65), (True, True, 35, 56, 59)),
        (120, (True, True, 26, 40, 44), (True, True, 24, 40, 46)),
        (100, (False, True, 0, 46, 44), (False, True, 0, 44, 46)),
        (80, (False, False, 0, 0, 70), (False, False, 0, 0, 70)),
        (60, (False, False, 0, 0, 50), (False, False, 0, 0, 50)),
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
        140,
        AdaptiveReaderLayoutPreferences(),
        MEDIA_PROFILE,
        priority=priority,  # type: ignore[arg-type]
    )

    assert getattr(layout, f"{priority}_open") is True
    assert layout.priority_pane is None


def test_shared_resolution_preserves_hysteresis() -> None:
    preferences = AdaptiveReaderLayoutPreferences()
    collapsed = resolve_adaptive_reader_layout(117, preferences, MEDIA_PROFILE)
    boundary = resolve_adaptive_reader_layout(118, preferences, MEDIA_PROFILE, previous=collapsed)
    still_collapsed = resolve_adaptive_reader_layout(121, preferences, MEDIA_PROFILE, previous=boundary)
    reopened = resolve_adaptive_reader_layout(122, preferences, MEDIA_PROFILE, previous=boundary)

    assert boundary.library_open is False
    assert still_collapsed.library_open is False
    assert reopened.library_open is True


@pytest.mark.parametrize("pane,width", [("library", 120), ("items", 96)])
def test_explicit_reopen_is_not_delayed_by_resize_hysteresis(pane, width):
    preferences = AdaptiveReaderLayoutPreferences(library_open=pane == "library")
    previous = resolve_media_reader_layout(width - 4, preferences)
    assert not getattr(previous, f"{pane}_open")
    reopened = resolve_media_reader_layout(width, preferences, previous=previous, priority=pane)
    assert getattr(reopened, f"{pane}_open")


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
    [(116, True), (100, True), (98, True), (97, False), (80, False), (60, False)],
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
        ("not-a-number", 36),
        (True, 36),
        (None, 36),
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


#: task-32127: Notes joined Media on `list_grows`. At 235 columns its list
#: stayed pinned at the 40-cell target beside a Reader holding 151 columns
#: of "Select a note to edit it here.", which is what clipped the titles,
#: the ages and the delete receipt's Undo.
LIST_GROWTH_PROFILE_NAMES = {"LIBRARY_NOTES_READER_PROFILE"}


def test_only_the_media_profile_opts_into_list_growth() -> None:
    assert MEDIA_READER_LAYOUT_PROFILE.list_grows is True
    assert MEDIA_READER_LAYOUT_PROFILE.grip_width == PANE_GRIP_WIDTH == 5
    for name, profile in DECLARED_PROFILES.items():
        assert profile.grip_width == PANE_GRIP_WIDTH, name
    assert AdaptiveReaderLayoutProfile().grip_width == PANE_GRIP_WIDTH
    assert set(SIBLING_PROFILES.values()) <= set(DECLARED_PROFILES.values())
    assert len(DECLARED_PROFILES) >= 6, sorted(DECLARED_PROFILES)
    for name, profile in DECLARED_PROFILES.items():
        assert profile.list_grows is (name in LIST_GROWTH_PROFILE_NAMES), name
    assert LIST_GROWTH_PROFILE_NAMES <= set(DECLARED_PROFILES)
    assert AdaptiveReaderLayoutProfile().list_grows is False


@pytest.mark.parametrize("width", [100, 235])
def test_every_profile_reserves_exactly_the_grip_columns_it_paints(width: int) -> None:
    """task-31951 AC#3 / task-31952 AC#3: one number, not two.

    The resolver holds back ``2 * profile.grip_width`` AND reports that same
    per-grip width on the layout it returns, and the shell paints its grips
    from the layout it is mounted with (pinned on the rendered widgets in
    ``Tests/UI/test_library_adaptive_reader_shell.py``). A destination can no
    longer reserve five columns and paint one.
    """
    profiles = dict(DECLARED_PROFILES)
    profiles["MEDIA_READER_LAYOUT_PROFILE"] = MEDIA_READER_LAYOUT_PROFILE

    for name, profile in profiles.items():
        layout = resolve_adaptive_reader_layout(
            width, AdaptiveReaderLayoutPreferences(), profile
        )

        assert layout.grip_width == profile.grip_width, name
        assert (
            layout.library_width
            + layout.items_width
            + layout.reader_width
            + 2 * layout.grip_width
        ) == width, name


@pytest.mark.parametrize(
    ("surface", "width", "expected"),
    [
        # Preserve the pre-increase collapse boundaries with five-cell controls.
        ("conversations", 100, (False, True, 0, 46, 44)),
        ("conversations", 117, (False, True, 0, 56, 51)),
        ("conversations", 118, (True, True, 24, 40, 44)),
        ("conversations", 119, (True, True, 25, 40, 44)),
        ("conversations", 122, (True, True, 28, 40, 44)),
        ("conversations", 235, (True, True, 39, 50, 136)),
        ("skills", 100, (False, True, 0, 42, 48)),
        ("skills", 121, (False, True, 0, 56, 55)),
        ("skills", 122, (True, True, 24, 40, 48)),
        ("skills", 123, (True, True, 25, 40, 48)),
        ("skills", 126, (True, True, 28, 40, 48)),
        ("skills", 235, (True, True, 39, 50, 136)),
        ("collections", 100, (False, True, 0, 42, 48)),
        ("collections", 121, (False, True, 0, 56, 55)),
        ("collections", 122, (True, True, 24, 40, 48)),
        ("collections", 123, (True, True, 25, 40, 48)),
        ("collections", 126, (True, True, 28, 40, 48)),
        ("collections", 235, (True, True, 39, 50, 136)),
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
        (100, (False, True, 0, 44, 46)),
        (119, (False, True, 0, 56, 53)),
        (120, (True, True, 24, 40, 46)),
        (121, (True, True, 25, 40, 46)),
        (127, (True, True, 29, 42, 46)),
    ],
)
def test_media_layout_across_the_rail_open_threshold(
    width: int, expected: tuple[bool, bool, int, int, int]
) -> None:
    layout = resolve_media_reader_layout(width, MediaReaderLayoutPreferences())

    assert _pane_widths(layout) == expected


@pytest.mark.parametrize(
    ("width", "expected"),
    [
        # At 235 the list reaches its comfort ceiling; 139 is the first
        # open-rail width with two surplus cells, shared between both panes.
        (235, (True, True, 39, 56, 130)),
        (139, (True, True, 31, 51, 47)),
    ],
)
def test_media_items_column_grows_once_the_reader_is_comfortable(
    width: int, expected: tuple[bool, bool, int, int, int]
) -> None:
    layout = resolve_media_reader_layout(width, MediaReaderLayoutPreferences())

    assert _pane_widths(layout) == expected


@pytest.mark.parametrize("custom_items_width", [32, 34, 48])
@pytest.mark.parametrize("width", [160, 235])
def test_a_typed_custom_items_width_is_obeyed_by_the_growth_gate(
    width: int, custom_items_width: int
) -> None:
    """Settings > Appearance > Custom widths is a hand-typed number.

    "Automatic" adapts; "Custom" obeys. Without the `list_grows` gate the
    typed value was silently overridden above ~130 columns (review
    Important 1).

    Scope: the `list_grows` gate only, with BOTH panes open. It does NOT
    cover the two comfort clamps, which still widen a typed width once the
    Library pane is gone -- see
    `test_a_typed_custom_items_width_is_still_widened_once_the_library_closes`
    for what those actually do, and why.
    """
    custom = MediaReaderLayoutPreferences(
        custom_widths_enabled=True,
        library_width=31,
        items_width=custom_items_width,
    )

    grown = resolve_media_reader_layout(width, custom)
    ungrown = resolve_adaptive_reader_layout(
        width, custom, MEDIA_PROFILE_WITHOUT_GROWTH
    )

    assert grown.items_width == custom_items_width
    assert grown == ungrown


@pytest.mark.parametrize("priority", [None, "items"])
def test_a_typed_custom_items_width_is_still_widened_once_the_library_closes(
    priority: str | None,
) -> None:
    """task-31953: the comfort clamps are NOT gated on custom widths.

    Decision: documented, not changed. TWO clamps widen a typed width once
    the Library pane is gone -- the library-closed clamp in
    `adaptive_reader_state.py` (`if items_open and not library_open`) and the
    priority-pane clamp above it -- so "obey the typed width" is not the
    one-line change this test-debt rider is scoped to, and
    `test_resolution_never_mutates_saved_preferences` already pins one of the
    widened values (a typed 40 resolves to 56). Changing it is a
    user-visible width change on all four reader surfaces; it needs its own
    task. This pin records what the resolver does today so the next reader
    change is not blamed for it.
    """
    custom = MediaReaderLayoutPreferences(
        custom_widths_enabled=True,
        library_width=31,
        items_width=32,
    )

    layout = resolve_media_reader_layout(100, custom, priority=priority)

    # A typed 32 paints 44: 100 cells less the two five-cell grips and the
    # 46-cell Reader minimum.
    assert (layout.library_open, layout.items_width) == (False, 44)


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


def test_empty_reader_gives_freed_columns_to_the_items_list_at_235() -> None:
    """task-31979: with no item open the Reader shows only its placeholder.

    The width it would otherwise reserve for a document is wasted while the
    Items list truncates long titles, so that width goes to the list instead,
    down to the Reader's own floor. Opening an item (reader_has_item=True)
    restores the split unchanged.
    """
    prefs = MediaReaderLayoutPreferences()
    with_item = resolve_media_reader_layout(235, prefs, reader_has_item=True)
    no_item = resolve_media_reader_layout(235, prefs, reader_has_item=False)

    # Item open: exactly the pre-task split (do not regress it).
    assert _pane_widths(with_item) == (True, True, 39, 56, 130)
    # No item open: the Items pane absorbs the freed Reader columns down to
    # the Reader's floor, so a 98-char title stops truncating at ~56 cells.
    assert no_item.reader_width == MEDIA_READER_LAYOUT_PROFILE.work_min_width
    assert no_item.items_width == 140
    assert no_item.items_width > with_item.items_width
    # Both panes and the two grips still tile the full terminal width.
    assert (
        no_item.library_width
        + no_item.items_width
        + no_item.reader_width
        + 2 * MEDIA_READER_LAYOUT_PROFILE.grip_width
    ) == 235


@pytest.mark.parametrize("width", [60, 80, 100, 120, 160, 235])
def test_reader_has_item_defaults_to_the_pre_task_behavior(width: int) -> None:
    """task-31979: the new parameter defaults to True, so every existing
    caller and every item-open resolution is byte-for-byte unchanged."""
    prefs = MediaReaderLayoutPreferences()
    assert resolve_media_reader_layout(width, prefs) == resolve_media_reader_layout(
        width, prefs, reader_has_item=True
    )


def test_empty_reader_widening_is_off_under_custom_widths() -> None:
    """task-31979: Custom widths obey the typed number (like the list_grows
    gate); only Automatic mode adapts to the empty Reader."""
    custom = MediaReaderLayoutPreferences(
        custom_widths_enabled=True, library_width=31, items_width=40
    )
    assert resolve_media_reader_layout(
        235, custom, reader_has_item=False
    ) == resolve_media_reader_layout(235, custom, reader_has_item=True)


@pytest.mark.parametrize("bad_value", [1, 0, "True", None])
def test_reader_has_item_rejects_a_non_boolean(bad_value) -> None:
    """task-32039 AC#3: the boundary arg is bool-validated like its siblings.

    ``reader_has_item`` fed a truthiness check without a type guard, so ``1``
    (or any truthy non-bool) silently passed. It is validated now, matching the
    ``width``/``preferences``/``profile`` guards.

    Args:
        bad_value: A non-boolean the resolver must refuse.
    """
    with pytest.raises(TypeError, match="reader_has_item must be a boolean"):
        resolve_adaptive_reader_layout(
            235,
            AdaptiveReaderLayoutPreferences(),
            MEDIA_PROFILE,
            reader_has_item=bad_value,
        )


# -- task-32389: the items priority does not outrank task-32065's rule ----


@pytest.mark.parametrize(
    "profile",
    [
        pytest.param(MEDIA_READER_LAYOUT_PROFILE, id="media"),
        pytest.param(screen_constants.LIBRARY_NOTES_READER_PROFILE, id="notes"),
    ],
)
@pytest.mark.parametrize("width", [48, 56, 60, 63])
def test_an_items_priority_below_the_floor_still_gives_an_empty_stage_to_the_list(
    profile, width: int
) -> None:
    """task-32389: both profiles that opt into the rule, not just Notes.

    The starved priority branch returns before task-32065's rescue, so an
    ``items`` priority used to pin the list at its 32-cell floor and hand the
    rest to a work pane with nothing in it. Media reaches that state in
    production too (the Trash view resolves with ``priority="items"`` and no
    item open), which is why the drop is claimed for both profiles rather
    than scoped to Notes -- review finding F2.
    """
    layout = resolve_adaptive_reader_layout(
        width,
        AdaptiveReaderLayoutPreferences(),
        profile,
        priority="items",
        reader_has_item=False,
    )

    preferences = AdaptiveReaderLayoutPreferences()
    assert layout.items_open is True
    # The list takes its preferred width, or everything the grips leave when
    # that is narrower -- never the 32-cell floor with the rest parked in an
    # empty pane. What is left over is below the work pane's own minimum, so
    # nothing usable is being held back.
    assert layout.items_width == min(
        preferences.items_width, width - 2 * profile.grip_width
    ), layout
    assert layout.reader_width < profile.work_min_width, (
        f"{layout.reader_width} columns are still held by an empty work pane "
        f"while the list is cut to {layout.items_width}."
    )
    if width == 60:
        # The width the critique ran at, pinned exactly (task-32389 AC#1).
        assert (layout.items_width, layout.reader_width) == (50, 0)


@pytest.mark.parametrize(
    "profile",
    [
        pytest.param(MEDIA_READER_LAYOUT_PROFILE, id="media"),
        pytest.param(screen_constants.LIBRARY_NOTES_READER_PROFILE, id="notes"),
    ],
)
@pytest.mark.parametrize("width", [48, 56, 60, 63])
def test_an_items_priority_still_reopens_a_closed_list_below_the_floor(
    profile, width: int
) -> None:
    """Review finding F3: the priority is the pane's only reopen lever.

    task-32065's rescue branch requires ``preferences.items_open``, so
    dropping the priority for a CLOSED pane discarded the reopen request and
    left the stage to the empty work pane. Latent in production (every caller
    flips the preference before resolving) and pinned here because this is a
    shared resolver.
    """
    layout = resolve_adaptive_reader_layout(
        width,
        AdaptiveReaderLayoutPreferences(items_open=False),
        profile,
        priority="items",
        reader_has_item=False,
    )

    assert layout.items_open is True
    assert layout.items_width >= profile.list_min_width


@pytest.mark.parametrize("width", [48, 56, 60, 63])
def test_an_items_priority_is_untouched_once_the_work_pane_has_something(
    width: int,
) -> None:
    """The drop is gated on an EMPTY work pane; a read item resolves as before."""
    for profile in (
        MEDIA_READER_LAYOUT_PROFILE,
        screen_constants.LIBRARY_NOTES_READER_PROFILE,
    ):
        layout = resolve_adaptive_reader_layout(
            width,
            AdaptiveReaderLayoutPreferences(),
            profile,
            priority="items",
            reader_has_item=True,
        )
        assert layout.priority_pane == "items"
        assert layout.reader_width > 0
