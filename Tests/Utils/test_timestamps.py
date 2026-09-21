"""The shared UTC timestamp helper (TASK-32803.1).

Pins the canonical stored shape, the assume-UTC contract for naive values, and
that every timestamp shape currently on disk round-trips through the parser.
Gate-free (Tests/Utils): pure stdlib datetime, no storage admission.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tldw_chatbook.Utils.timestamps import (
    is_canonical_utc,
    parse_utc,
    to_utc_iso,
    utc_now,
    utc_now_iso,
)

pytestmark = pytest.mark.unit


def test_canonical_shape_is_fixed_width_millisecond_z():
    now = utc_now_iso()
    assert now.endswith("Z")
    assert len(now) == 24  # YYYY-MM-DDTHH:MM:SS.mmmZ
    # fixed width + Z means TEXT (lexical) order == chronological order
    earlier = to_utc_iso(datetime(2026, 9, 20, 12, 0, 0, 0))
    later = to_utc_iso(datetime(2026, 9, 20, 12, 0, 1, 0))
    assert earlier < later


def test_utc_now_is_aware_utc():
    assert utc_now().utcoffset() == timedelta(0)


def test_to_utc_iso_treats_naive_as_utc_and_converts_aware():
    naive = datetime(2026, 9, 20, 12, 34, 56, 789000)
    assert to_utc_iso(naive) == "2026-09-20T12:34:56.789Z"
    est = timezone(timedelta(hours=-5))
    aware = datetime(2026, 9, 20, 7, 34, 56, 789000, tzinfo=est)
    assert to_utc_iso(aware) == "2026-09-20T12:34:56.789Z"


@pytest.mark.parametrize(
    "shape",
    [
        "2026-09-20T12:34:56.789Z",  # canonical
        "2026-09-20T12:34:56.789012+00:00",  # micros + offset
        "2026-09-20T12:34:56.789012",  # naive micros
        "2026-09-20 12:34:56",  # SQLite CURRENT_TIMESTAMP
        "2026-09-20 12:34",  # %Y-%m-%d %H:%M
        "2026-09-20",  # date only
        "2026-09-20T12:34:56Z",  # seconds + Z
        "2026-09-20 12:34:56.789Z",  # space + millis + Z
        "  2026-09-20T12:34:56.789Z  ",  # surrounding whitespace
    ],
)
def test_parse_utc_accepts_every_shape_on_disk_as_aware_utc(shape):
    parsed = parse_utc(shape)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timedelta(0)


def test_canonical_string_round_trips():
    now = utc_now_iso()
    assert to_utc_iso(parse_utc(now)) == now


@pytest.mark.parametrize("bad", ["", "   ", "not a date", "2026-13-40"])
def test_parse_utc_rejects_garbage(bad):
    with pytest.raises(ValueError):
        parse_utc(bad)


def test_is_canonical_utc_accepts_only_the_canonical_shape():
    assert is_canonical_utc(utc_now_iso())
    assert is_canonical_utc("2026-09-20T12:34:56.789Z")
    # everything the writer must NOT emit
    for bad in [
        "2026-09-20T12:34:56.789012Z",   # microseconds
        "2026-09-20T12:34:56.789+00:00",  # offset, not Z
        "2026-09-20T12:34:56Z",           # no fraction
        "2026-09-20 12:34:56.789Z",       # space separator
        "2026-09-20T12:34:56.789z",       # lowercase z
        "",
        None,
        12345,
    ]:
        assert not is_canonical_utc(bad), bad


def test_everything_to_utc_iso_emits_is_canonical():
    from datetime import datetime, timezone, timedelta
    for dt in [
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 9, 20, 12, 34, 56, 789012, tzinfo=timezone.utc),
        datetime(2026, 9, 20, 7, 0, 0, tzinfo=timezone(timedelta(hours=-5))),
        datetime(2026, 9, 20, 12, 0, 0),  # naive -> UTC
    ]:
        assert is_canonical_utc(to_utc_iso(dt))
