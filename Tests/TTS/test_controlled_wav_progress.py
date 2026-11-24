"""Bounded statistics and absolute source-frame clocks."""

from tldw_chatbook.TTS.controlled_wav_progress import ProgressClock, StatsReader


def record(value):
    return f"{value} M-A: 0.000 fd= 0 aq= 0KB".encode()


def test_parser_framing_rejects_oversize_and_incomplete_records():
    reader = StatsReader()
    assert reader.feed(record("1.25")[:9]) == []
    assert reader.feed(record("1.25")[9:] + b"\r\n") == [1.25]
    assert reader.feed(
        b" " * 1025 + record("8") + b"\n" + record("-1") + b"\n" + record("2") + b"\r"
    ) == [2]


def test_clock_excludes_jumps_repeats_and_stale_observations():
    clock = ProgressClock(48000, 96000, 7, 10.0)
    for seconds, stamp, generation in [
        (1.0, 9.9, 7),
        (1.0, 10.0, 6),
        (1.9, 10.0, 7),
        (float("nan"), 10.0, 7),
    ]:
        clock.observe(seconds, stamp, generation)
    assert clock.position(10.0) is None
    clock.observe(0.98, 10.01, 7)
    assert clock.position(10.01).frame == 48000
    clock.observe(1.05, 10.07, 7)
    clock.observe(1.05, 10.55, 7)
    clock.observe(0.99, 10.56, 7)
    assert clock.position(10.58) is None
    assert clock.expired(10.58)


def test_terminal_requires_advancement_elapsed_and_fresh_physical_padding():
    clock = ProgressClock(9600, 14400, 8, 3.0)
    clock.observe(0.20, 3.01, 8)
    clock.observe(0.39, 3.20, 8)
    assert not clock.natural_candidate(3.20)
    clock.observe(0.42, 3.23, 8)
    assert clock.position(3.23).frame == 14400
    assert clock.natural_candidate(3.23)
    assert not clock.natural_candidate(4.00)


def test_late_or_future_record_does_not_poison_last_trustworthy_bookmark():
    clock = ProgressClock(0, 48000, 1, 5.0)
    clock.observe(0.0, 5.01, 1, now_s=5.01)
    clock.observe(0.05, 5.06, 1, now_s=5.70)
    clock.observe(0.08, 5.08, 1, now_s=5.02)
    assert clock.position(5.09).frame == 0


def test_adjacent_jump_cannot_cross_content_end_inside_cumulative_skew_bound():
    clock = ProgressClock(0, 14400, 3, 0)
    for source, stamp in [(0, 0), (0.001, 0.249), (0.499, 0.251)]:
        clock.observe(source, stamp, 3)
    assert clock.position(0.301).frame == 48
    assert not clock.natural_candidate(0.301)
