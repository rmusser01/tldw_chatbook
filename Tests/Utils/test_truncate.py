"""The shared truncation helper (TASK-32808.3).

One truncator replacing the scattered `text[:n] + "..."` re-rolls and the
`_truncate*`/`_ellipsize` copies. Default marker is the single-char ellipsis
"…" (owner decision 2026-09-20); `marker=` overrides it for ASCII/terminal
contexts. The result including the marker never exceeds the limit.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Utils.Utils import truncate

pytestmark = pytest.mark.unit

ELLIPSIS = "…"


def test_passthrough_none_empty_and_within_limit():
    assert truncate(None, 10) is None
    assert truncate("", 10) == ""
    assert truncate("short", 10) == "short"
    assert truncate("exactly10!", 10) == "exactly10!"  # == limit, unchanged


def test_default_marker_is_single_char_ellipsis_and_result_fits_limit():
    out = truncate("a" * 20, 10)
    assert out == "a" * 9 + ELLIPSIS
    assert len(out) == 10


def test_ascii_marker_override():
    out = truncate("a" * 20, 10, marker="...")
    assert out == "a" * 7 + "..."
    assert len(out) == 10


def test_small_limits():
    # marker fits whenever limit > len(marker): 1 char of text + the 1-char marker
    assert truncate("abcdef", 2) == "a" + ELLIPSIS
    # limit <= len(marker): hard cut, no room for the marker
    assert truncate("abcdef", 1) == "a"
    assert truncate("abcdef", 0) == ""
    # a 3-char ASCII marker needs limit > 3 to appear
    assert truncate("abcdef", 3, marker="...") == "abc"  # hard cut, marker can't fit
    assert truncate("abcdef", 4, marker="...") == "a..."


def test_marker_length_budget_is_respected_for_multichar_markers():
    # a 3-char marker reserves 3; total never exceeds limit
    assert len(truncate("x" * 50, 12, marker="...")) == 12
    assert len(truncate("x" * 50, 12)) == 12  # 1-char default
