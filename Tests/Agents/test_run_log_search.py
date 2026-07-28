# Tests/Agents/test_run_log_search.py
"""Search: literal by default, structured filters, bounded regex."""

from tldw_chatbook.Agents.run_log_format import RunLogRecord
from tldw_chatbook.Agents.run_log_search import (
    MAX_REGEX_SCAN_CHARS,
    format_results,
    search_records,
)


def rec(number, content, **kw):
    base = dict(
        number=number,
        run_id="r",
        kind="primary",
        type="tool_result",
        ts="t",
        content=content,
    )
    base.update(kw)
    return RunLogRecord(**base)


CORPUS = [
    rec(1, "opened the config file", tool="read_file", status="ok"),
    rec(2, "connection refused", tool="web_search", status="error"),
    rec(3, "thinking about it", type="model"),
    rec(4, "wrote the config file", tool="write_file", status="ok"),
]


def test_literal_contains_is_the_default_and_is_not_a_regex():
    assert [r.number for r in search_records(CORPUS, contains="config file")] == [1, 4]
    # A regex metacharacter is matched literally, never compiled.
    assert search_records(CORPUS, contains="config.file") == []


def test_literal_search_is_unbounded_by_line_length():
    long_record = [rec(9, "x" * 5000 + "NEEDLE")]
    assert len(search_records(long_record, contains="NEEDLE")) == 1


def test_structured_filters_compose():
    hits = search_records(CORPUS, status="error")
    assert [r.number for r in hits] == [2]
    assert [r.number for r in search_records(CORPUS, type="model")] == [3]
    assert [r.number for r in search_records(CORPUS, tool="write_file")] == [4]


def test_record_range_slices():
    assert [r.number for r in search_records(CORPUS, from_record=3)] == [3, 4]
    assert [r.number for r in search_records(CORPUS, to_record=2)] == [1, 2]


def test_context_returns_neighbours_in_order_without_duplicates():
    hits = search_records(CORPUS, contains="refused", context=1)
    assert [r.number for r in hits] == [1, 2, 3]


def test_negative_context_is_clamped_and_still_returns_the_hit():
    # Reviewer finding: a negative context made low > high, so
    # range(low, high + 1) came back empty and even the matching record
    # itself was silently dropped -- "No matching records." even though a
    # match exists. context must clamp to 0, not widen the window the
    # wrong direction.
    hits = search_records(CORPUS, contains="refused", context=-5)
    assert [r.number for r in hits] == [2]


def test_regex_mode_is_opt_in_and_scan_bounded():
    assert [r.number for r in search_records(CORPUS, pattern=r"conn\w+")] == [2]
    # Beyond the scan window the pattern cannot match, by design.
    far = [rec(9, "y" * (MAX_REGEX_SCAN_CHARS + 50) + "NEEDLE")]
    assert search_records(far, pattern="NEEDLE") == []
    assert len(search_records(far, contains="NEEDLE")) == 1


def test_invalid_regex_returns_no_hits_rather_than_raising():
    assert search_records(CORPUS, pattern="(unclosed") == []


def test_limit_caps_results():
    assert len(search_records(CORPUS, limit=2)) == 2


def test_format_results_is_readable_and_truncates_long_content():
    text = format_results([rec(7, "z" * 900, tool="read_file")], max_chars=50)
    assert "record 000007" in text
    assert "read_file" in text
    assert len(text) < 300


def test_format_results_reports_no_matches():
    assert "no matching records" in format_results([]).lower()


def test_match_beyond_max_chars_is_rendered_not_silently_dropped():
    """TASK-1250 live reproduction.

    Observed against a real llama.cpp model with max_tool_result_chars=500
    and a 2,925-character tool result whose answer sat at character 2,646:
    `format_results` always rendered `content[:max_chars]` from offset 0 --
    the same ceiling that truncated the result in history in the first
    place -- so a record far larger than that ceiling rendered byte-
    identical to the truncated view the model already saw. Worse,
    `contains=` could legitimately MATCH the record (search_records has no
    scan bound for literal search) and then render a body that did not
    contain the match: the agent was told the record matched and shown
    text that contradicted it. The model followed the trailer, searched by
    content, and produced an EMPTY final answer -- it had the information
    in its log and could not reach it.

    Before the fix: format_results had no `contains` parameter at all, so
    this call fails outright. After the fix: the render must be centred on
    the match, so the marker -- 2,146 characters past the render ceiling --
    must be visible in the rendered text.
    """
    marker = "ZEBRA_9931"
    content = "x" * 2646 + marker + "y" * 269
    assert len(content) == 2925  # pins the live evidence's exact shape
    record = rec(3, content)

    # The record DOES match -- search_records has no scan bound for
    # literal `contains`, so this half of the bug was never in question.
    hits = search_records([record], contains=marker)
    assert [r.number for r in hits] == [3]

    # The RENDER must contain what the search says matched.
    rendered = format_results(hits, max_chars=500, contains=marker)
    assert marker in rendered


def test_offset_pages_through_a_large_record_to_its_end():
    # A record several times max_chars, with a marker reachable only by
    # following the render's own "Use offset=N to continue" pointer.
    tail = "TAIL_MARKER_END"
    content = "m" * 1985 + tail
    assert len(content) == 2000
    record = rec(5, content)

    first = format_results([record], max_chars=500)
    assert tail not in first
    assert "Use offset=500 to continue" in first

    second = format_results([record], max_chars=500, offset=500)
    assert tail not in second
    assert "Use offset=1000 to continue" in second

    third = format_results([record], max_chars=500, offset=1000)
    assert tail not in third
    assert "Use offset=1500 to continue" in third

    fourth = format_results([record], max_chars=500, offset=1500)
    assert tail in fourth
    # The final page reaches the record's true end: no further pointer.
    assert "Use offset=" not in fourth


def test_offset_negative_or_past_end_is_clamped_not_empty():
    content = "n" * 2000
    record = rec(6, content)

    # A negative offset (a model guessing at search_run_log's args could
    # send one) must behave like offset=0, never raise, never render an
    # empty window.
    negative = format_results([record], max_chars=500, offset=-999)
    body = negative.split("\n", 1)[1]
    assert body.startswith("n" * 500)

    # An offset past the record's end must still render its final window
    # rather than nothing.
    past_end = format_results([record], max_chars=500, offset=999_999)
    body = past_end.split("\n", 1)[1]
    assert body.strip() != ""
    assert "chars 1500-2000 of 2000" in past_end


def test_no_query_render_still_starts_at_offset_zero():
    content = "p" * 2000
    record = rec(9, content)
    rendered = format_results([record], max_chars=500)
    body = rendered.split("\n", 1)[1]
    assert body.startswith("p" * 500)


def test_limit_applies_to_hits_before_context_expansion():
    # Reproducer from reviewer: 11-record corpus with only one match at record 6,
    # context=3, limit=3. The match must be in the result.
    corpus = [
        rec(1, "a"),
        rec(2, "b"),
        rec(3, "c"),
        rec(4, "d"),
        rec(5, "e"),
        rec(6, "NEEDLE"),
        rec(7, "f"),
        rec(8, "g"),
        rec(9, "h"),
        rec(10, "i"),
        rec(11, "j"),
    ]
    result = search_records(corpus, contains="NEEDLE", context=3, limit=3)
    # The one matching record (6) must be present.
    assert 6 in [r.number for r in result]
    # Context should include records 3-9 (3 before and after record 6),
    # which is 7 records total (exceeding limit because context is additional).
    assert [r.number for r in result] == [3, 4, 5, 6, 7, 8, 9]
