"""B27: the think filter counts UTF-8 bytes at slice level, not per character.

The pre-change ``StartAnchoredThinkSplitter._bounded_thinking`` visited every
codepoint in Python (``ord(text[index])`` plus a per-character add) -- O(n)
Python-level operations per chunk. The rewrite encodes each slice once
(``len(text[start:end].encode("utf-8"))``) and finds surrogates with a
module-level compiled pattern.

Golden equivalence: a verbatim copy of the old function (below) acts as the
oracle over a corpus covering ASCII, 2/3/4-byte characters, astral
characters, ``surrogateescape`` bytes, multi-range calls, and both failure
modes (surrogate present; byte cap exceeded).
"""

from __future__ import annotations

import builtins

import pytest

from tldw_chatbook.Chat.thinking_blocks import MAX_THINKING_TEXT_BYTES
from tldw_chatbook.Chat.llamacpp_think_filter import StartAnchoredThinkSplitter


def _legacy_bounded_thinking(
    thinking_bytes: int, ranges: list[tuple[str, int, int]]
) -> tuple[int, str] | None:
    """Verbatim pre-B27 ``_bounded_thinking`` body, as the golden oracle.

    Returns ``(new_total, joined)`` on success and ``None`` on the terminal
    capture failure (surrogate present or byte cap exceeded) -- matching the
    old method, which reset ``_thinking_bytes`` to 0 on failure.
    """
    total = thinking_bytes
    for text, start, end in ranges:
        for index in range(start, end):
            codepoint = ord(text[index])
            if 0xD800 <= codepoint <= 0xDFFF:
                return None
            total += (
                1
                if codepoint <= 0x7F
                else 2
                if codepoint <= 0x7FF
                else 3
                if codepoint <= 0xFFFF
                else 4
            )
            if total > MAX_THINKING_TEXT_BYTES:
                return None
    return total, "".join(text[start:end] for text, start, end in ranges)


_CORPUS: list[str] = [
    "",
    "plain ascii text",
    "2-byte accents: héllo wörld, café naïve",
    "3-byte CJK: 日本語テキスト、中文测试",
    "4-byte astral: math \U0001d54a, emoji \U0001f984 \U0001f600",
    "ZWJ family: \U0001f468\u200d\U0001f469\u200d\U0001f467\u200d\U0001f466",
    b"\xff\xfe".decode("utf-8", "surrogateescape"),
    "mixed ascii + \U0001f984 + 日本 + café",
    "<think>thinking</think>visible",
]


def _cases() -> list[tuple[int, list[tuple[str, int, int]]]]:
    """(starting _thinking_bytes, ranges) cases incl. boundary failures."""
    cases: list[tuple[int, list[tuple[str, int, int]]]] = []
    for text in _CORPUS:
        cases.append((0, [(text, 0, len(text))]))
        # sliced ranges: skip heads/tails, empty ranges
        if len(text) >= 4:
            cases.append((0, [(text, 1, len(text) - 1)]))
            cases.append((0, [(text, 2, 2), (text, 2, len(text))]))
    # accumulated-state cases across two ranges of different texts
    cases.append((0, [("日本語", 0, 3), ("ascii", 0, 5)]))
    cases.append((10, [("café", 0, 4), ("\U0001f984", 0, 1)]))
    # byte-cap failure: start just under the cap with a multi-byte tail
    near_cap = MAX_THINKING_TEXT_BYTES - 1
    cases.append((near_cap, [("日本", 0, 2)]))  # +6 bytes -> over cap
    cases.append((near_cap, [("a", 0, 1)]))  # exactly at cap -> success
    cases.append((MAX_THINKING_TEXT_BYTES, [("a", 0, 1)]))  # +1 -> over cap
    # surrogate failure (lone surrogate from surrogateescape)
    lone = b"\xff".decode("utf-8", "surrogateescape")
    cases.append((0, [("ok" + lone + "tail", 0, 8)]))
    cases.append((0, [("clean", 0, 5), (lone, 0, 1)]))
    return cases


@pytest.mark.parametrize("start_bytes,ranges", _cases())
def test_bounded_thinking_matches_legacy_oracle(
    start_bytes: int, ranges: list[tuple[str, int, int]]
) -> None:
    splitter = StartAnchoredThinkSplitter()
    splitter._thinking_bytes = start_bytes
    result = splitter._bounded_thinking(*ranges)

    oracle = _legacy_bounded_thinking(start_bytes, ranges)

    if oracle is None:
        assert result is None
        assert splitter._thinking_bytes == 0, "failure must reset the counter"
    else:
        expected_total, expected_text = oracle
        assert result == expected_text
        assert splitter._thinking_bytes == expected_total


def test_bounded_thinking_does_not_visit_characters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rewrite must count bytes per slice, not per codepoint."""
    ord_calls = {"count": 0}
    real_ord = builtins.ord

    def counting_ord(value):
        ord_calls["count"] += 1
        return real_ord(value)

    monkeypatch.setattr(builtins, "ord", counting_ord)

    splitter = StartAnchoredThinkSplitter()
    chunk = "ascii + 日本 + \U0001f984 padding" * 20  # ~500 chars, mixed widths
    splitter._bounded_thinking((chunk, 0, len(chunk)))

    assert ord_calls["count"] == 0, (
        f"_bounded_thinking visited {ord_calls['count']} codepoints in "
        "Python; byte counting must happen at slice level"
    )


@pytest.mark.parametrize("used_bytes", [0, MAX_THINKING_TEXT_BYTES - 1])
def test_oversized_thinking_rejects_before_full_utf8_allocation(used_bytes):
    import tracemalloc

    text = "x" * (MAX_THINKING_TEXT_BYTES * 4)
    splitter = StartAnchoredThinkSplitter()
    splitter._thinking_bytes = used_bytes
    tracemalloc.start()
    try:
        result = splitter._bounded_thinking((text, 0, len(text)))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert result is None
    assert splitter._thinking_bytes == 0
    assert peak < MAX_THINKING_TEXT_BYTES * 2
