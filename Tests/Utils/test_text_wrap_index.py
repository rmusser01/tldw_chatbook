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
