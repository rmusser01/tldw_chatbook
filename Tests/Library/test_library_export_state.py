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
    MEDIA_QUALITY_OPTIONS,
    build_library_export_form_state,
    default_export_name,
    next_media_quality,
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
        counts={"media": 128, "conversations": 542, "notes": 87},
        name="x",
        description="",
        media_quality=DEFAULT_MEDIA_QUALITY,
        destination="/tmp/out.zip",
    )
    assert state.counts_loading is False
    assert state.scope_line == "Everything: 128 media · 542 conversations · 87 notes"


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


@pytest.mark.parametrize("kind", ["conversations", "notes"])
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


# --- next_media_quality -------------------------------------------------------


def test_next_media_quality_cycles_and_wraps():
    assert next_media_quality("thumbnail") == "compressed"
    assert next_media_quality("compressed") == "original"
    assert next_media_quality("original") == "thumbnail"
    assert next_media_quality("bogus") == MEDIA_QUALITY_OPTIONS[0]


# --- normalize_export_destination ---------------------------------------------


def test_normalize_destination_appends_zip_suffix_when_absent():
    assert normalize_export_destination(PurePath("/tmp/foo")) == PurePath("/tmp/foo.zip")


def test_normalize_destination_replaces_a_different_suffix():
    assert normalize_export_destination(PurePath("/tmp/foo.txt")) == PurePath("/tmp/foo.zip")


def test_normalize_destination_leaves_zip_suffix_untouched_case_insensitive():
    assert normalize_export_destination(PurePath("/tmp/foo.ZIP")) == PurePath("/tmp/foo.ZIP")
    assert normalize_export_destination(PurePath("/tmp/foo.zip")) == PurePath("/tmp/foo.zip")
