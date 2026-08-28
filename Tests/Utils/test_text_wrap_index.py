"""Tests for exact wrap arithmetic (TASK-22500)."""

import random

import pytest
from rich.console import Console
from rich.text import Text

from tldw_chatbook.Utils.text_wrap_index import WrapIndex, divide_source_line


def test_divide_source_line_agrees_with_public_text_wrap():
    """Pins the private rich._wrap.divide_line against the public API.

    If a Rich upgrade moves or changes divide_line, this fails loudly here
    instead of silently changing how every document wraps.
    """
    console = Console(width=40)
    lines = [
        "short",
        "the quick brown fox jumps over the lazy dog and keeps on running forever",
        "supercalifragilisticexpialidocious " * 3,
        "",
    ]
    for line in lines:
        ours = len(divide_source_line(line, 40)) + 1
        theirs = max(1, len(Text(line).wrap(console, 40)))
        assert ours == theirs, f"row count disagreed for {line!r}"


def test_index_maps_rows_to_lines_and_segments():
    lines = ["a" * 10, "b" * 25, "c"]
    index = WrapIndex.build(lines, width=10)
    assert index.virtual_height == 1 + 3 + 1
    assert index.row_to_line(0) == (0, 0)
    assert index.row_to_line(1) == (1, 0)
    assert index.row_to_line(3) == (1, 2)
    assert index.row_to_line(4) == (2, 0)
    assert index.line_start_row(2) == 4


def test_segments_round_trip_the_source_line():
    index = WrapIndex.build(["hello world " * 5], width=17)
    assert "".join(index.segments(0)) == "hello world " * 5


def test_empty_document_has_one_row():
    index = WrapIndex.build([""], width=20)
    assert index.virtual_height == 1
    assert index.row_to_line(0) == (0, 0)
    assert index.segments(0) == [""]


def test_exact_index_beats_character_division_on_ragged_text():
    """The cheap approximation was measured wrong on 12.4% of ragged lines.

    Character division (cell_len // width) drifts virtual height ~2.9%,
    which is a visibly wrong scrollbar and a match jump that lands in the
    wrong place. This pins that WrapIndex is exact where that is not.
    """
    from rich.cells import cell_len

    random.seed(7)
    words = ["a", "to", "the", "quick", "extraordinarily", "fox", "internationalization"]
    lines = [
        " ".join(random.choice(words) for _ in range(random.randint(20, 60)))
        for _ in range(500)
    ]
    console = Console(width=100)
    index = WrapIndex.build(lines, width=100)
    exact = sum(max(1, len(Text(line).wrap(console, 100))) for line in lines)
    approx = sum(max(1, -(-cell_len(line) // 100)) for line in lines)
    assert index.virtual_height == exact
    assert approx != exact, "fixture no longer exercises the approximation's error"


def test_divide_source_line_fallback_matches_private_api(monkeypatch):
    """Monkeypatch away rich._wrap.divide_line to test the fallback path.

    The fallback uses the public Text.wrap API to achieve the same row counts
    as the private divide_line function. This test forces the fallback by
    setting the module's _rich_divide_line to None.
    """
    import tldw_chatbook.Utils.text_wrap_index as wrap_module

    # Force the fallback by disabling the private API
    monkeypatch.setattr(wrap_module, "_rich_divide_line", None)

    # Re-run the agreement test with the fallback
    console = Console(width=40)
    lines = [
        "short",
        "the quick brown fox jumps over the lazy dog and keeps on running forever",
        "supercalifragilisticexpialidocious " * 3,
        "",
    ]
    for line in lines:
        ours = len(divide_source_line(line, 40)) + 1
        theirs = max(1, len(Text(line).wrap(console, 40)))
        assert ours == theirs, f"fallback row count disagreed for {line!r}"


# Whitespace-heavy on purpose: Text.wrap rstrips each divided line, so any
# fallback that sums the RENDERED segment lengths under-counts by exactly the
# stripped run and drifts further with every wrap point. The drift is invisible
# to a row-count comparison -- the counts still match -- which is why the two
# agreement tests above could not see it.
_OFFSET_FIXTURES = [
    ("hello world  foo bar baz", 5),
    ("aaa   bbb   ccc   ddd", 4),
    ("the quick brown fox jumps over the lazy dog", 10),
    ("supercalifragilisticexpialidocious", 5),
    ("   leading spaces here", 3),
    ("trailing spaces   ", 4),
    ("a b", 1),
    ("one", 10),
    ("", 5),
    ("word " * 30, 13),
    ("日本語 wide text here", 6),
]


def test_divide_source_line_offsets_match_the_private_api_exactly():
    """Row counts are not enough: the OFFSETS have to match.

    A fallback that agreed on counts while disagreeing on offsets produced a
    9-character segment at width 4, which adjust_cell_length then truncated --
    silently dropping text off the end of the document.
    """
    from rich._wrap import divide_line

    for line, width in _OFFSET_FIXTURES:
        assert divide_source_line(line, width) == list(divide_line(line, width)), (
            f"offsets disagreed for {line!r} at width {width}"
        )


def test_the_fallback_never_loses_text_or_outruns_its_width(monkeypatch):
    """The fallback's real contract.

    Exact offset parity is NOT achievable through the public API: Rich
    absorbs the whitespace after a WORD break into the preceding segment but
    starts a new segment after a hard mid-word fold, and ``Text.wrap``
    erases that distinction (both arrive rstripped). Matching it exactly
    would mean reimplementing ``divide_line``.

    So the fallback is best-effort on where lines break, and strict on what
    must never happen: no character may be lost, and no segment may carry
    real content past the width -- that combination is what silently dropped
    text off the end of the document when the offsets drifted.
    """
    import tldw_chatbook.Utils.text_wrap_index as wrap_module

    monkeypatch.setattr(wrap_module, "_rich_divide_line", None)
    for line, width in _OFFSET_FIXTURES:
        index = WrapIndex.build([line], width)
        segments = index.segments(0)
        assert "".join(segments) == line, f"text lost for {line!r} at width {width}"
        for segment in segments:
            assert len(segment.rstrip()) <= width, (
                f"segment {segment!r} outruns width {width} for line {line!r}"
            )
        assert index.virtual_height == len(segments)


def test_no_segment_outruns_its_width_on_the_fallback_path(monkeypatch):
    """The failure mode the offset drift actually caused: an oversized
    trailing segment that gets truncated rather than wrapped."""
    import tldw_chatbook.Utils.text_wrap_index as wrap_module

    monkeypatch.setattr(wrap_module, "_rich_divide_line", None)
    index = WrapIndex.build(["aaa   bbb   ccc   ddd"], width=4)
    segments = index.segments(0)
    assert "".join(segments) == "aaa   bbb   ccc   ddd"
    # Trailing whitespace may legitimately ride along past the width (Rich
    # keeps it attached and rstrips at render); real CONTENT may not.
    for segment in segments:
        assert len(segment.rstrip()) <= 4, f"segment {segment!r} outruns width 4"


def test_segment_start_matches_a_running_sum():
    lines = ["alpha beta gamma delta " * 8, "short", "x" * 97]
    index = WrapIndex.build(lines, width=11)
    for line_index in range(len(lines)):
        segments = index.segments(line_index)
        running = 0
        for segment_index, segment in enumerate(segments):
            assert index.segment_start(line_index, segment_index) == running
            running += len(segment)
    # Out-of-range indices clamp rather than raise.
    assert index.segment_start(0, 10_000) == index.segment_start(0, len(index.segments(0)) - 1)


def test_short_ascii_lines_never_reach_rich(monkeypatch):
    """The index build runs on the UI thread, so it must stay far below the
    repo's 100 ms worker threshold.

    A short pure-ASCII line cannot wrap, so it must not pay a Rich
    ``divide_line`` call. Asserted by CALL COUNT, not wall clock (the 15457
    probe rule): the measured effect is a 2.5 MB document building in 2.5 ms
    instead of 140 ms.
    """
    import tldw_chatbook.Utils.text_wrap_index as wrap_module

    calls = {"n": 0}
    real = wrap_module._rich_divide_line

    def counting(line, width):
        calls["n"] += 1
        return real(line, width)

    monkeypatch.setattr(wrap_module, "_rich_divide_line", counting)

    ascii_doc = [f"line {i} of ordinary short ascii text" for i in range(5000)]
    index = WrapIndex.build(ascii_doc, width=80)
    assert index.virtual_height == 5000
    assert calls["n"] == 0, f"short ASCII lines cost {calls['n']} divide_line calls"

    # Anything that CAN wrap still goes through Rich: a long line, and a
    # short line whose glyphs may be wider than one cell each.
    calls["n"] = 0
    WrapIndex.build(["x" * 500, "日本語のテキスト"], width=80)
    assert calls["n"] == 2


def test_wide_glyph_lines_still_wrap_by_cell_width():
    """The fast path must not swallow a line that is short in CHARACTERS but
    too wide in CELLS -- CJK glyphs are 2 cells each."""
    line = "日本語" * 20  # 60 characters, 120 cells
    index = WrapIndex.build([line], width=40)
    assert index.virtual_height > 1, "wide-glyph line must still wrap"
    assert "".join(index.segments(0)) == line
