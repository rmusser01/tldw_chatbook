"""Pure display-state contract for the Library notes sync panel."""

from tldw_chatbook.Library.library_notes_sync_state import (
    AUTO_SYNC_INTERVAL_SECONDS,
    SYNC_CONFLICTS,
    SYNC_DIRECTIONS,
    LibraryNotesSyncState,
    append_activity,
    auto_sync_label,
    count_noun,
    next_sync_conflict,
    next_sync_direction,
    sync_conflict_label,
    sync_direction_label,
    sync_status_line,
)


# ----- next_sync_direction: cycling + wrap + unknown -----------------------

def test_next_sync_direction_cycles_in_order():
    assert next_sync_direction("bidirectional") == "disk_to_db"
    assert next_sync_direction("disk_to_db") == "db_to_disk"


def test_next_sync_direction_wraps_to_first():
    assert next_sync_direction("db_to_disk") == "bidirectional"


def test_next_sync_direction_unknown_value_returns_first():
    assert next_sync_direction("bogus") == SYNC_DIRECTIONS[0]
    assert next_sync_direction("") == SYNC_DIRECTIONS[0]


# ----- next_sync_conflict: cycling + wrap + unknown -------------------------
# "ask" is gone from the cycle entirely (A1): the engine keeps its own
# ConflictResolution.ASK, but this panel never offers it -- a dead
# affordance, since nothing in the Library ever prompts on a conflict.

def test_next_sync_conflict_cycles_in_order():
    assert next_sync_conflict("newer_wins") == "disk_wins"
    assert next_sync_conflict("disk_wins") == "db_wins"
    assert next_sync_conflict("db_wins") == "newer_wins"


def test_next_sync_conflict_wraps_to_first():
    assert next_sync_conflict("db_wins") == "newer_wins"


def test_next_sync_conflict_unknown_value_returns_first():
    assert next_sync_conflict("bogus") == SYNC_CONFLICTS[0]
    assert next_sync_conflict("") == SYNC_CONFLICTS[0]


def test_next_sync_conflict_never_offers_ask():
    assert "ask" not in SYNC_CONFLICTS
    cycled = {next_sync_conflict(value) for value in SYNC_CONFLICTS}
    assert "ask" not in cycled


# ----- labels: known values + raw fallback ----------------------------------
# B2: "DB" jargon is gone from the labels -- the config keys/enum values are
# unchanged, only the human-facing wording says "Library" instead.

def test_sync_direction_label_known_values():
    assert sync_direction_label("bidirectional") == "Bidirectional"
    assert sync_direction_label("disk_to_db") == "Disk → Library"
    assert sync_direction_label("db_to_disk") == "Library → Disk"


def test_sync_direction_label_unknown_falls_back_to_raw():
    assert sync_direction_label("mystery") == "mystery"


def test_sync_conflict_label_known_values():
    assert sync_conflict_label("newer_wins") == "Newer wins"
    assert sync_conflict_label("disk_wins") == "Disk wins"
    assert sync_conflict_label("db_wins") == "Library wins"


def test_sync_conflict_label_unknown_falls_back_to_raw():
    assert sync_conflict_label("mystery") == "mystery"
    # "ask" is no longer a recognized label -- it now falls back to raw,
    # same as any other unrecognized value.
    assert sync_conflict_label("ask") == "ask"


# ----- sync_status_line: all variants ---------------------------------------

def test_sync_status_line_idle():
    assert sync_status_line("idle") == "idle"


def test_sync_status_line_syncing_shows_progress_fraction():
    assert sync_status_line("syncing", processed=3, total=12) == "syncing · 3/12"


def test_sync_status_line_syncing_with_zero_total():
    assert sync_status_line("syncing", processed=0, total=0) == "syncing · 0/0"


def test_sync_status_line_done_shows_files_and_conflicts():
    assert sync_status_line("done", processed=12, total=12, conflicts=2) == (
        "done · 12 files · 2 conflicts"
    )


def test_sync_status_line_done_with_no_conflicts_omits_conflict_clause():
    assert sync_status_line("done", processed=12, total=12, conflicts=0) == "done · 12 files"


def test_sync_status_line_done_pluralizes_singular_file():
    assert sync_status_line("done", processed=1, total=1, conflicts=0) == "done · 1 file"


def test_sync_status_line_done_pluralizes_singular_conflict():
    assert sync_status_line("done", processed=1, total=1, conflicts=1) == (
        "done · 1 file · 1 conflict"
    )


def test_sync_status_line_done_zero_files_pluralizes():
    assert sync_status_line("done", processed=0, total=0, conflicts=0) == "done · 0 files"


def test_sync_status_line_failed_shows_reason():
    assert sync_status_line("failed", error="folder not found") == "failed · folder not found"


def test_sync_status_line_failed_with_no_reason():
    assert sync_status_line("failed", error="") == "failed"


def test_sync_status_line_unknown_status_falls_back_to_raw():
    assert sync_status_line("mystery") == "mystery"


# ----- append_activity: cap + ordering --------------------------------------

def test_append_activity_prepends_most_recent_first():
    lines = append_activity((), "first entry")
    lines = append_activity(lines, "second entry")
    assert lines == ("second entry", "first entry")


def test_append_activity_caps_at_default_20():
    lines: tuple[str, ...] = tuple(f"entry {i}" for i in range(20))
    result = append_activity(lines, "newest")
    assert len(result) == 20
    assert result[0] == "newest"
    assert result[-1] == "entry 18"
    assert "entry 19" not in result


def test_append_activity_respects_custom_cap():
    lines = ("a", "b", "c")
    result = append_activity(lines, "d", cap=3)
    assert result == ("d", "a", "b")


def test_append_activity_empty_start():
    result = append_activity((), "only")
    assert result == ("only",)


# ----- count_noun: pluralization helper -------------------------------------

def test_count_noun_singular_has_no_trailing_s():
    assert count_noun(1, "file") == "1 file"


def test_count_noun_plural_has_trailing_s():
    assert count_noun(2, "file") == "2 files"


def test_count_noun_zero_is_plural():
    assert count_noun(0, "file") == "0 files"


def test_count_noun_negative_is_plural():
    assert count_noun(-1, "file") == "-1 files"


# ----- auto_sync_label: cadence + state -------------------------------------

def test_auto_sync_label_enabled_shows_checkmark_and_cadence():
    assert auto_sync_label(True) == "auto-sync: every 5m ✓"


def test_auto_sync_label_disabled_shows_circle_and_cadence():
    assert auto_sync_label(False) == "auto-sync: every 5m ○"


def test_auto_sync_label_cadence_matches_interval_constant():
    minutes = AUTO_SYNC_INTERVAL_SECONDS // 60
    assert f"every {minutes}m" in auto_sync_label(True)


# ----- LibraryNotesSyncState.running: default + explicit ---------------------

def test_library_notes_sync_state_running_defaults_false():
    state = LibraryNotesSyncState(
        folder="~/Notes",
        direction="bidirectional",
        conflict="newer_wins",
        auto_sync=False,
        status_line="idle",
        activity_lines=(),
    )
    assert state.running is False


def test_library_notes_sync_state_running_can_be_set_true():
    state = LibraryNotesSyncState(
        folder="~/Notes",
        direction="bidirectional",
        conflict="newer_wins",
        auto_sync=False,
        status_line="syncing · 0/0",
        activity_lines=(),
        running=True,
    )
    assert state.running is True
