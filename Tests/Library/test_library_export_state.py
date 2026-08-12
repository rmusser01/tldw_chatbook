"""Library export canvas display-state contracts.

Covers ``tldw_chatbook.Library.library_export_state``: the pure form-state
builder the export canvas (Task 2) renders from, plus its small pure
helpers (quality cycling, destination ``.zip`` normalization). Filesystem
and DB reads (whether a destination already exists, the full-query counts
themselves) are the screen's job -- every fact this module needs arrives
as a plain argument, never read here.
"""

from __future__ import annotations

from datetime import date
from pathlib import PurePath

import pytest

from tldw_chatbook.Library.library_export_scope import ExportScope
from tldw_chatbook.Library.library_export_state import (
    DEFAULT_MEDIA_QUALITY,
    EMPTY_SCOPE_COPY,
    EXPORT_BUTTON_COUNTING_TOOLTIP,
    EXPORT_BUTTON_NO_DESTINATION_TOOLTIP,
    EXPORT_BUTTON_READY_TOOLTIP,
    EXPORT_BUTTON_RUNNING_TOOLTIP,
    MEDIA_QUALITY_OPTIONS,
    build_library_export_form_state,
    default_export_name,
    export_button_tooltip,
    format_last_export_line,
    media_quality_helper_copy,
    normalize_export_destination,
)


# --- default_export_name -----------------------------------------------------


def test_default_export_name_stamps_todays_local_date():
    assert default_export_name(date(2026, 7, 11)) == "Library export 2026-07-11"


# --- build_library_export_form_state: counts loading / landed ---------------


def test_counts_none_renders_counting_placeholder_and_disables_export():
    state = build_library_export_form_state(
        scope=ExportScope(kind="everything"),
        counts=None,
        name="Library export 2026-07-11",
        description="",
        media_quality=DEFAULT_MEDIA_QUALITY,
        destination="/tmp/out.zip",
    )
    assert state.counts_loading is True
    assert state.scope_line == "Counting…"
    assert state.export_enabled is False


def test_counts_landed_renders_scope_label():
    state = build_library_export_form_state(
        scope=ExportScope(kind="everything"),
        counts={"media": 128, "conversations": 542, "notes": 87, "prompts": 13},
        name="x",
        description="",
        media_quality=DEFAULT_MEDIA_QUALITY,
        destination="/tmp/out.zip",
    )
    assert state.counts_loading is False
    assert state.scope_line == (
        "Everything: 128 media · 542 conversations · 87 notes · 13 prompts"
    )


def test_media_scoped_label_carries_type_filter():
    state = build_library_export_form_state(
        scope=ExportScope(kind="media", media_type="video"),
        counts={"media": 12},
        name="x",
        description="",
        media_quality=DEFAULT_MEDIA_QUALITY,
        destination="",
    )
    assert state.scope_line == "Media (type: video) · 12 items"


# --- Export button gating ----------------------------------------------------


def test_export_disabled_until_destination_chosen():
    state = build_library_export_form_state(
        scope=ExportScope(kind="everything"),
        counts={"media": 1, "conversations": 0, "notes": 0},
        name="x",
        description="",
        media_quality=DEFAULT_MEDIA_QUALITY,
        destination="",
    )
    assert state.export_enabled is False


def test_export_enabled_once_counts_landed_total_positive_and_destination_set():
    state = build_library_export_form_state(
        scope=ExportScope(kind="everything"),
        counts={"media": 1, "conversations": 0, "notes": 0},
        name="x",
        description="",
        media_quality=DEFAULT_MEDIA_QUALITY,
        destination="/tmp/out.zip",
    )
    assert state.export_enabled is True


def test_export_disabled_while_running_even_with_everything_else_satisfied():
    state = build_library_export_form_state(
        scope=ExportScope(kind="everything"),
        counts={"media": 1, "conversations": 0, "notes": 0},
        name="x",
        description="",
        media_quality=DEFAULT_MEDIA_QUALITY,
        destination="/tmp/out.zip",
        running=True,
    )
    assert state.export_enabled is False


def test_empty_scope_disables_export_and_shows_helper_copy():
    state = build_library_export_form_state(
        scope=ExportScope(kind="notes"),
        counts={"media": 0, "conversations": 0, "notes": 0},
        name="x",
        description="",
        media_quality=DEFAULT_MEDIA_QUALITY,
        destination="/tmp/out.zip",
    )
    assert state.export_enabled is False
    assert state.empty_scope_line == EMPTY_SCOPE_COPY


def test_nonempty_scope_never_shows_empty_scope_line():
    state = build_library_export_form_state(
        scope=ExportScope(kind="notes"),
        counts={"media": 0, "conversations": 0, "notes": 3},
        name="x",
        description="",
        media_quality=DEFAULT_MEDIA_QUALITY,
        destination="/tmp/out.zip",
    )
    assert state.empty_scope_line == ""


def test_empty_scope_line_withheld_while_counts_still_loading():
    state = build_library_export_form_state(
        scope=ExportScope(kind="notes"),
        counts=None,
        name="x",
        description="",
        media_quality=DEFAULT_MEDIA_QUALITY,
        destination="/tmp/out.zip",
    )
    assert state.empty_scope_line == ""


# --- show_media_fields --------------------------------------------------------


@pytest.mark.parametrize("kind", ["everything", "media"])
def test_media_bearing_scopes_show_media_fields(kind):
    state = build_library_export_form_state(
        scope=ExportScope(kind=kind),
        counts={"media": 1},
        name="x",
        description="",
        media_quality=DEFAULT_MEDIA_QUALITY,
        destination="",
    )
    assert state.show_media_fields is True


@pytest.mark.parametrize("kind", ["conversations", "notes", "prompts"])
def test_non_media_scopes_hide_media_fields(kind):
    state = build_library_export_form_state(
        scope=ExportScope(kind=kind),
        counts={},
        name="x",
        description="",
        media_quality=DEFAULT_MEDIA_QUALITY,
        destination="",
    )
    assert state.show_media_fields is False


# --- Overwrite confirm line ---------------------------------------------------


def test_overwrite_line_empty_when_destination_does_not_exist():
    state = build_library_export_form_state(
        scope=ExportScope(kind="everything"),
        counts={"media": 1},
        name="x",
        description="",
        media_quality=DEFAULT_MEDIA_QUALITY,
        destination="/tmp/out.zip",
        destination_exists=False,
    )
    assert state.overwrite_line == ""


def test_overwrite_line_names_the_destination_file_when_it_exists():
    state = build_library_export_form_state(
        scope=ExportScope(kind="everything"),
        counts={"media": 1},
        name="x",
        description="",
        media_quality=DEFAULT_MEDIA_QUALITY,
        destination="/tmp/out.zip",
        destination_exists=True,
    )
    assert state.overwrite_line == "Overwrites out.zip"


def test_overwrite_line_empty_when_no_destination_chosen_yet():
    state = build_library_export_form_state(
        scope=ExportScope(kind="everything"),
        counts={"media": 1},
        name="x",
        description="",
        media_quality=DEFAULT_MEDIA_QUALITY,
        destination="",
        destination_exists=True,
    )
    assert state.overwrite_line == ""


# --- running / status / error passthrough ------------------------------------


def test_running_status_and_error_lines_pass_through_unchanged():
    state = build_library_export_form_state(
        scope=ExportScope(kind="everything"),
        counts={"media": 1},
        name="x",
        description="",
        media_quality=DEFAULT_MEDIA_QUALITY,
        destination="/tmp/out.zip",
        running=True,
        status_line="Exporting… (1 items)",
        error_line="",
    )
    assert state.running is True
    assert state.status_line == "Exporting… (1 items)"
    assert state.error_line == ""

    failed_state = build_library_export_form_state(
        scope=ExportScope(kind="everything"),
        counts={"media": 1},
        name="x",
        description="",
        media_quality=DEFAULT_MEDIA_QUALITY,
        destination="/tmp/out.zip",
        error_line="Permission denied",
    )
    assert failed_state.error_line == "Permission denied"


# --- quality option set -------------------------------------------------------
# task-14902: ``next_media_quality`` retired with the per-press cycle -- the
# quality control now opens a direct-pick strip over MEDIA_QUALITY_OPTIONS
# (see Tests/UI/test_library_choice_strips.py), so the option tuple itself
# is the remaining contract.


def test_media_quality_options_are_the_three_known_values_in_order():
    assert MEDIA_QUALITY_OPTIONS == ("thumbnail", "compressed", "original")
    assert DEFAULT_MEDIA_QUALITY == "thumbnail"


def test_export_form_state_carries_quality_choices_visible_flag():
    state = build_library_export_form_state(
        scope=ExportScope(kind="everything"),
        counts={"media": 1, "conversations": 0, "notes": 0},
        name="Library export 2026-08-11",
        description="",
        media_quality="thumbnail",
        destination="",
        quality_choices_visible=True,
    )
    assert state.quality_choices_visible is True


# --- media_quality_helper_copy -------------------------------------------------
# task-2859 item 3: the helper line used to be ONE fixed sentence describing
# "original" quality, shown under the quality button no matter which value
# was actually selected -- so "quality: thumbnail ⇄" was captioned with
# "original copies full media files into the zip". Each option now gets its
# own honest caption naming its own effect.


def test_media_quality_helper_copy_matches_each_option_to_its_own_effect():
    assert "small preview" in media_quality_helper_copy("thumbnail")
    assert "shrinks" in media_quality_helper_copy("compressed")
    assert "full media files" in media_quality_helper_copy("original")
    # A thumbnail caption never claims to keep full files, and vice versa.
    assert "full media files" not in media_quality_helper_copy("thumbnail")


def test_media_quality_helper_copy_degrades_to_original_for_unknown_value():
    assert media_quality_helper_copy("bogus") == media_quality_helper_copy("original")


# --- normalize_export_destination ---------------------------------------------


def test_normalize_destination_appends_zip_suffix_when_absent():
    assert normalize_export_destination(PurePath("/tmp/foo")) == PurePath(
        "/tmp/foo.zip"
    )


def test_normalize_destination_replaces_a_different_suffix():
    assert normalize_export_destination(PurePath("/tmp/foo.txt")) == PurePath(
        "/tmp/foo.zip"
    )


def test_normalize_destination_leaves_zip_suffix_untouched_case_insensitive():
    assert normalize_export_destination(PurePath("/tmp/foo.ZIP")) == PurePath(
        "/tmp/foo.ZIP"
    )
    assert normalize_export_destination(PurePath("/tmp/foo.zip")) == PurePath(
        "/tmp/foo.zip"
    )


# --- export_button_tooltip: task-2858 AC#3 (LIB-11) --------------------------


def _state(**overrides):
    base = dict(
        scope=ExportScope(kind="everything"),
        counts={"media": 1, "conversations": 0, "notes": 0},
        name="x",
        description="",
        media_quality=DEFAULT_MEDIA_QUALITY,
        destination="/tmp/out.zip",
    )
    base.update(overrides)
    return build_library_export_form_state(**base)


def test_tooltip_is_the_ready_hint_when_export_is_enabled():
    state = _state()
    assert state.export_enabled is True
    assert export_button_tooltip(state) == EXPORT_BUTTON_READY_TOOLTIP


def test_tooltip_names_running_as_the_blocker_while_running():
    state = _state(running=True)
    assert export_button_tooltip(state) == EXPORT_BUTTON_RUNNING_TOOLTIP


def test_tooltip_names_counting_as_the_blocker_before_counts_land():
    state = _state(counts=None)
    assert export_button_tooltip(state) == EXPORT_BUTTON_COUNTING_TOOLTIP


def test_tooltip_reuses_the_empty_scope_copy_verbatim():
    """The disabled reason must match the on-canvas "Nothing to export in
    this scope." line exactly -- not a second, potentially-drifting
    string -- per task-2858 AC#3's "same predicate" requirement."""
    state = _state(counts={"media": 0, "conversations": 0, "notes": 0})
    assert state.empty_scope_line == EMPTY_SCOPE_COPY
    assert export_button_tooltip(state) == EMPTY_SCOPE_COPY


def test_tooltip_names_missing_destination_as_the_blocker():
    state = _state(destination="")
    assert export_button_tooltip(state) == EXPORT_BUTTON_NO_DESTINATION_TOOLTIP


# --- format_last_export_line: task-2858 AC#3 (LIB-12) ------------------------


def test_last_export_line_empty_before_any_export_this_session():
    assert format_last_export_line("", 0.0, now=1000.0) == ""


def test_last_export_line_reads_just_now_within_the_first_minute():
    line = format_last_export_line("/tmp/out.zip", 970.0, now=1000.0)
    assert line == "Last export: /tmp/out.zip · just now"


def test_last_export_line_reads_minutes_ago():
    line = format_last_export_line("/tmp/out.zip", 1000.0, now=1000.0 + 5 * 60)
    assert line == "Last export: /tmp/out.zip · 5m ago"


def test_last_export_line_reads_hours_ago():
    line = format_last_export_line("/tmp/out.zip", 1000.0, now=1000.0 + 3 * 3600)
    assert line == "Last export: /tmp/out.zip · 3h ago"


def test_last_export_line_reads_days_ago():
    line = format_last_export_line("/tmp/out.zip", 1000.0, now=1000.0 + 2 * 86400)
    assert line == "Last export: /tmp/out.zip · 2d ago"


def test_build_library_export_form_state_passes_last_export_line_through():
    state = _state(last_export_line="Last export: /tmp/prior.zip · 1h ago")
    assert state.last_export_line == "Last export: /tmp/prior.zip · 1h ago"


def test_build_library_export_form_state_defaults_last_export_line_empty():
    state = _state()
    assert state.last_export_line == ""
