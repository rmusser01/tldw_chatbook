"""Pure display-state contract for the Library notes sync panel."""

from tldw_chatbook.Library.library_notes_sync_state import (
    SYNC_CONFLICTS,
    SYNC_DIRECTIONS,
    append_activity,
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

def test_next_sync_conflict_cycles_in_order():
    assert next_sync_conflict("newer_wins") == "disk_wins"
    assert next_sync_conflict("disk_wins") == "db_wins"
    assert next_sync_conflict("db_wins") == "ask"


def test_next_sync_conflict_wraps_to_first():
    assert next_sync_conflict("ask") == "newer_wins"


def test_next_sync_conflict_unknown_value_returns_first():
    assert next_sync_conflict("bogus") == SYNC_CONFLICTS[0]
    assert next_sync_conflict("") == SYNC_CONFLICTS[0]


# ----- labels: known values + raw fallback ----------------------------------

def test_sync_direction_label_known_values():
    assert sync_direction_label("bidirectional") == "Bidirectional"
    assert sync_direction_label("disk_to_db") == "Disk → DB"
    assert sync_direction_label("db_to_disk") == "DB → Disk"


def test_sync_direction_label_unknown_falls_back_to_raw():
    assert sync_direction_label("mystery") == "mystery"


def test_sync_conflict_label_known_values():
    assert sync_conflict_label("newer_wins") == "Newer wins"
    assert sync_conflict_label("disk_wins") == "Disk wins"
    assert sync_conflict_label("db_wins") == "DB wins"
    assert sync_conflict_label("ask") == "Ask"


def test_sync_conflict_label_unknown_falls_back_to_raw():
    assert sync_conflict_label("mystery") == "mystery"


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
