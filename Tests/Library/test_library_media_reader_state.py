"""Pure contracts for the Library Media reader session and layout."""

from __future__ import annotations

import pytest

from tldw_chatbook.Library.library_media_reader_state import (
    ITEMS_MAX_WIDTH,
    ITEMS_MIN_WIDTH,
    ITEMS_TARGET_WIDTH,
    LAYOUT_HYSTERESIS_WIDTH,
    LIBRARY_MAX_WIDTH,
    LIBRARY_MIN_WIDTH,
    LIBRARY_TARGET_WIDTH,
    PANE_GRIP_WIDTH,
    READER_COMFORT_WIDTH,
    SELECTION_SETTLE_SECONDS,
    LibraryMediaReaderSessionState,
    MediaReaderLayoutPreferences,
    begin_selection,
    enter_external_detail,
    leave_external_detail,
    normalize_media_reader_preferences,
    resolve_media_reader_layout,
    set_mode,
    settle_failure,
    settle_success,
)


def test_default_preferences_are_both_open_and_fixed() -> None:
    assert normalize_media_reader_preferences({}) == MediaReaderLayoutPreferences(
        library_open=True,
        items_open=True,
        custom_widths_enabled=False,
        library_width=LIBRARY_TARGET_WIDTH,
        items_width=ITEMS_TARGET_WIDTH,
    )


@pytest.mark.parametrize(
    ("library_width", "items_width", "expected_library", "expected_items"),
    [
        (1, 1, LIBRARY_MIN_WIDTH, ITEMS_MIN_WIDTH),
        (999, 999, LIBRARY_MAX_WIDTH, ITEMS_MAX_WIDTH),
        ("36", "60", 36, 60),
        (None, object(), LIBRARY_TARGET_WIDTH, ITEMS_TARGET_WIDTH),
    ],
)
def test_custom_widths_clamp_to_declared_minimums_and_maximums(
    library_width: object,
    items_width: object,
    expected_library: int,
    expected_items: int,
) -> None:
    preferences = normalize_media_reader_preferences(
        {
            "custom_widths_enabled": True,
            "library_width": library_width,
            "items_width": items_width,
        }
    )

    assert preferences.library_width == expected_library
    assert preferences.items_width == expected_items


def test_fixed_mode_ignores_saved_custom_width_values() -> None:
    preferences = normalize_media_reader_preferences(
        {
            "library_open": "false",
            "items_open": "true",
            "custom_widths_enabled": False,
            "library_width": LIBRARY_MAX_WIDTH,
            "items_width": ITEMS_MAX_WIDTH,
        }
    )

    assert preferences == MediaReaderLayoutPreferences(
        library_open=False,
        items_open=True,
        custom_widths_enabled=False,
        library_width=LIBRARY_TARGET_WIDTH,
        items_width=ITEMS_TARGET_WIDTH,
    )


def test_responsive_collapse_does_not_mutate_preferences() -> None:
    preferences = normalize_media_reader_preferences({})

    layout = resolve_media_reader_layout(80, preferences)

    assert layout.library_open is False
    assert layout.items_open is False
    assert preferences.library_open is True
    assert preferences.items_open is True


@pytest.mark.parametrize(
    ("width", "library_open", "items_open"),
    [(160, True, True), (120, False, True), (80, False, False)],
)
def test_normal_resolution_collapses_library_then_items(
    width: int, library_open: bool, items_open: bool
) -> None:
    layout = resolve_media_reader_layout(width, normalize_media_reader_preferences({}))

    assert (layout.library_open, layout.items_open) == (
        library_open,
        items_open,
    )
    assert (
        layout.library_width
        + layout.items_width
        + layout.reader_width
        + 2 * PANE_GRIP_WIDTH
        == width
    )


@pytest.mark.parametrize(
    ("priority", "expected_library_width", "expected_items_width"),
    [("library", LIBRARY_MIN_WIDTH, 0), ("items", 0, ITEMS_MIN_WIDTH)],
)
def test_explicit_open_collapses_other_pane_first_and_uses_requested_minimum(
    priority: str,
    expected_library_width: int,
    expected_items_width: int,
) -> None:
    preferences = normalize_media_reader_preferences({f"{priority}_open": False})

    layout = resolve_media_reader_layout(
        80,
        preferences,
        priority=priority,  # type: ignore[arg-type]
    )

    assert layout.library_width == expected_library_width
    assert layout.items_width == expected_items_width
    assert layout.priority_pane == priority


def test_reader_can_drop_below_comfort_after_explicit_open_without_overflow() -> None:
    layout = resolve_media_reader_layout(
        60,
        normalize_media_reader_preferences({"items_open": False}),
        priority="items",
    )

    assert layout.items_open is True
    assert layout.items_width == ITEMS_MIN_WIDTH
    assert layout.reader_width == 60 - 2 * PANE_GRIP_WIDTH - ITEMS_MIN_WIDTH
    assert layout.reader_width < READER_COMFORT_WIDTH


def test_two_grips_and_reader_remain_reachable_at_sixty_columns() -> None:
    layout = resolve_media_reader_layout(60, normalize_media_reader_preferences({}))

    assert layout.library_width == 0
    assert layout.items_width == 0
    assert layout.reader_width == 50
    assert layout.reader_width + 2 * PANE_GRIP_WIDTH == 60


def test_returning_width_restores_target_widths_not_intermediate_widths() -> None:
    preferences = normalize_media_reader_preferences({})
    narrow = resolve_media_reader_layout(80, preferences, priority="items")

    wide = resolve_media_reader_layout(160, preferences, previous=narrow)

    assert wide.library_width == LIBRARY_TARGET_WIDTH
    assert wide.items_width == ITEMS_TARGET_WIDTH
    assert wide.priority_pane is None


def test_explicit_open_priority_survives_narrow_resize_resolution() -> None:
    preferences = normalize_media_reader_preferences({})
    opened = resolve_media_reader_layout(80, preferences, priority="items")

    resized = resolve_media_reader_layout(81, preferences, previous=opened)

    assert resized.items_open is True
    assert resized.items_width == ITEMS_MIN_WIDTH
    assert resized.priority_pane == "items"


def test_hysteresis_prevents_one_column_resize_thrashing() -> None:
    preferences = normalize_media_reader_preferences({})
    collapsed = resolve_media_reader_layout(121, preferences)
    boundary = resolve_media_reader_layout(122, preferences, previous=collapsed)
    reopened = resolve_media_reader_layout(
        122 + LAYOUT_HYSTERESIS_WIDTH,
        preferences,
        previous=boundary,
    )

    assert collapsed.library_open is False
    assert boundary.library_open is False
    assert reopened.library_open is True
    assert reopened.library_width == LIBRARY_TARGET_WIDTH


def test_shrink_expand_cycles_are_idempotent() -> None:
    preferences = normalize_media_reader_preferences({})
    widths = (160, 120, 80, 120, 160)

    def cycle(previous=None):
        layouts = []
        for width in widths:
            previous = resolve_media_reader_layout(
                width, preferences, previous=previous
            )
            layouts.append(previous)
        return layouts

    first = cycle()
    second = cycle(first[-1])

    assert first == second


def _loaded_local(
    backing_id: int, title: str = "Loaded"
) -> LibraryMediaReaderSessionState:
    pending = begin_selection(
        LibraryMediaReaderSessionState(),
        f"local:media:{backing_id}",
        backing_id,
        title,
        immediate=True,
    )
    assert pending.pending_request is not None
    return settle_success(
        pending,
        pending.pending_request.generation,
        pending.pending_request.requested_id,
    )


def test_begin_selection_updates_selected_before_loaded() -> None:
    loaded = _loaded_local(1, "Alpha")

    pending = begin_selection(loaded, "local:media:2", 2, "Beta")

    assert pending.selected_id == "local:media:2"
    assert pending.selected_backing_id == 2
    assert pending.loaded_id == "local:media:1"
    assert pending.loaded_backing_id == 1
    assert pending.pending_request is not None
    assert pending.pending_request.requested_id == "local:media:2"


def test_pending_banner_can_name_selected_and_loaded_titles() -> None:
    pending = begin_selection(_loaded_local(1, "Alpha"), "local:media:2", 2, "Beta")

    assert pending.pending_banner == (
        "Loading preview for “Beta”… showing “Alpha” until ready."
    )


def test_enter_can_force_immediate_load_generation() -> None:
    settling = begin_selection(
        LibraryMediaReaderSessionState(), "local:media:3", 3, "Gamma"
    )
    immediate = begin_selection(settling, "local:media:3", 3, "Gamma", immediate=True)

    assert settling.pending_request is not None
    assert settling.pending_request.delay_seconds == SELECTION_SETTLE_SECONDS
    assert immediate.pending_request is not None
    assert immediate.pending_request.delay_seconds == 0
    assert immediate.pending_request.generation == (
        settling.pending_request.generation + 1
    )


def test_only_matching_generation_and_backend_qualified_id_can_settle() -> None:
    pending = begin_selection(
        LibraryMediaReaderSessionState(), "local:media:7", 7, "Local seven"
    )
    assert pending.pending_request is not None
    generation = pending.pending_request.generation

    wrong_backend = settle_success(pending, generation, "server:media:7")
    settled = settle_success(pending, generation, "local:media:7")

    assert wrong_backend is pending
    assert settled.loaded_id == "local:media:7"
    assert settled.pending_request is None


def test_stale_success_and_stale_failure_are_rejected() -> None:
    first = begin_selection(
        LibraryMediaReaderSessionState(), "local:media:1", 1, "Alpha"
    )
    assert first.pending_request is not None
    current = begin_selection(first, "local:media:2", 2, "Beta")
    assert current.pending_request is not None

    stale_success = settle_success(
        current,
        first.pending_request.generation,
        current.pending_request.requested_id,
    )
    stale_failure = settle_failure(
        current,
        first.pending_request.generation,
        current.pending_request.requested_id,
        "old failure",
    )

    assert stale_success is current
    assert stale_failure is current
    assert current.error is None


def test_selected_and_loaded_can_differ_only_while_pending() -> None:
    pending = begin_selection(_loaded_local(1, "Alpha"), "local:media:2", 2, "Beta")

    assert pending.selected_id != pending.loaded_id
    assert pending.pending_request is not None
    with pytest.raises(ValueError, match="differ"):
        LibraryMediaReaderSessionState(
            selected_id="local:media:2",
            selected_backing_id=2,
            selected_title="Beta",
            loaded_id="local:media:1",
            loaded_backing_id=1,
            loaded_title="Alpha",
        )


@pytest.mark.parametrize(
    ("loaded_backing_id", "loaded_title"),
    [("7", "Seven"), (7, "Stale seven")],
)
def test_settled_identity_rejects_inconsistent_backing_or_title(
    loaded_backing_id: object, loaded_title: str
) -> None:
    with pytest.raises(ValueError, match="selected and loaded"):
        LibraryMediaReaderSessionState(
            selected_id="local:media:7",
            selected_backing_id=7,
            selected_title="Seven",
            loaded_id="local:media:7",
            loaded_backing_id=loaded_backing_id,  # type: ignore[arg-type]
            loaded_title=loaded_title,
        )


def test_mode_persists_when_new_item_settles() -> None:
    analysis = set_mode(_loaded_local(1), "analysis")
    pending = begin_selection(analysis, "local:media:2", 2, "Beta", immediate=True)
    assert pending.pending_request is not None

    settled = settle_success(
        pending,
        pending.pending_request.generation,
        pending.pending_request.requested_id,
    )

    assert settled.mode == "analysis"
    assert settled.loaded_id == "local:media:2"


def test_external_server_session_cannot_collide_with_local_id() -> None:
    local = _loaded_local(7, "Local seven")

    external = enter_external_detail(local, 7, "Server seven")

    assert external.external_detail is True
    assert external.selected_id == "server:media:7"
    assert external.loaded_id == "server:media:7"
    assert external.selected_id != local.selected_id
    assert external.selected_backing_id == local.selected_backing_id == 7

    left = leave_external_detail(external)
    assert left.external_detail is False
    assert left.selected_id is None
    assert left.loaded_id is None
    assert left.request_generation > external.request_generation
