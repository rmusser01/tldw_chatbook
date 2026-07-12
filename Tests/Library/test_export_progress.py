from tldw_chatbook.Library.export_progress import (
    ExportProgressThrottle, format_export_progress_line,
)


def test_throttle_emits_on_first_call_and_phase_change():
    t = ExportProgressThrottle(min_interval=1.0)
    assert t.should_emit("media", 1, 10, now=100.0) is True     # first ever
    assert t.should_emit("media", 2, 10, now=100.1) is False    # within interval, same phase
    assert t.should_emit("packaging", 1, 5, now=100.2) is True  # phase change flushes


def test_throttle_emits_on_final_tick_and_interval():
    t = ExportProgressThrottle(min_interval=1.0)
    t.should_emit("media", 1, 10, now=0.0)
    assert t.should_emit("media", 10, 10, now=0.05) is True     # current >= total flushes
    t2 = ExportProgressThrottle(min_interval=0.1)
    t2.should_emit("media", 1, 10, now=0.0)
    assert t2.should_emit("media", 2, 10, now=0.2) is True      # interval elapsed


def test_throttle_total_zero_does_not_bypass_time_throttle():
    # An indeterminate/skipped phase (total == 0) must NOT force an emit on
    # every tick via `current >= total` (1 >= 0) — it should fall back to the
    # time throttle, or it would flood the UI thread.
    t = ExportProgressThrottle(min_interval=1.0)
    assert t.should_emit("media", 0, 0, now=0.0) is True    # first ever emits
    assert t.should_emit("media", 1, 0, now=0.1) is False   # within interval -> throttled
    assert t.should_emit("media", 2, 0, now=1.2) is True    # interval elapsed -> emits


def test_format_progress_line():
    assert format_export_progress_line("media", 42, 318) == "Collecting media…  42/318"
    assert format_export_progress_line("packaging", 210, 540) == "Packaging archive…  210/540 files"
    assert format_export_progress_line("relationships", 1, 1) == "Resolving links…  1/1"
