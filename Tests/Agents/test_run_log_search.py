# Tests/Agents/test_run_log_search.py
"""Search: literal by default, structured filters, bounded regex."""

import time

import pytest

from tldw_chatbook.Agents.run_log_format import RunLogRecord
from tldw_chatbook.Agents.run_log_search import (
    DEFAULT_SLICE_WIDTH,
    MAX_REGEX_SCAN_CHARS,
    MAX_SLICE_RECORDS,
    MAX_STATS_GROUPS,
    RunLogSearchPatternRejected,
    RunLogSearchTimeout,
    compute_stats,
    format_results,
    format_slice,
    format_stats,
    search_records,
    slice_records,
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


# -- F6 (Qodo #6): a model-controlled regex must never hang the worker -------


@pytest.mark.parametrize(
    "pattern",
    [
        "(a+)+",
        "(a*)*",
        "(a+)*",
        "(a*)+",
        "(a+?)+",  # lazy variant, same shape
        "((a+)+)+",  # nested twice
        "prefix(a+)+suffix",  # not anchored at the pattern's start
    ],
)
def test_catastrophic_pattern_shapes_are_rejected_quickly(pattern):
    """Each of these is the textbook nested-quantifier shape named in the
    finding. The screen must reject BEFORE compiling/executing -- this
    itself proves nothing hung (a real hang would time the test out), and
    the message must point the model at `contains=` instead.
    """
    started = time.monotonic()
    with pytest.raises(RunLogSearchPatternRejected, match="contains"):
        search_records(CORPUS, pattern=pattern)
    # "quickly" -- a generous ceiling far below what an actual catastrophic
    # match against even a short string would take (fractions of a second
    # vs. an unreachable amount of time).
    assert time.monotonic() - started < 1.0


@pytest.mark.parametrize(
    "pattern",
    [
        r"\d{3}-\d{4}",
        "(abc)+",
        "a+b*",
        "(foo|bar)+",
        "[a-z]+",
        "(ab)+c*",
        "config.*file",
        r"(a+)(b+)",  # two quantified groups, but NOT nested -- must not trip
    ],
)
def test_ordinary_patterns_are_not_rejected(pattern):
    """The screen is conservative by design -- it must never flag a normal
    pattern, even ones with multiple quantified groups, as long as they are
    not the specific nested shape."""
    # Must not raise; whether it matches anything is irrelevant here.
    search_records(CORPUS, pattern=pattern)


def test_search_records_raises_a_clear_timeout_past_its_deadline():
    """F6 layer 1: a wall-clock deadline checked between records. Uses a
    near-zero deadline against a small corpus so the very first
    between-record check trips it deterministically -- this does not (and
    cannot, from pure Python) prove protection against a single hung regex
    evaluation; see the module docstring for that limitation.
    """
    with pytest.raises(RunLogSearchTimeout, match="wall-clock"):
        search_records(CORPUS, contains="config", deadline_seconds=0.0)


def test_search_records_completes_well_within_a_generous_deadline():
    # Regression guard: the deadline check itself must not be so aggressive
    # it breaks an ordinary, fast search.
    result = search_records(CORPUS, contains="config file", deadline_seconds=5.0)
    assert [r.number for r in result] == [1, 4]


# -- F7 (Qodo #7): a capped record must say so, not claim completeness -------


def test_format_results_flags_a_record_the_writer_itself_capped():
    record = rec(1, "y" * 50, truncated_from=1_000_000)
    out = format_results([record])
    assert "cannot be recovered" in out
    assert "1000000" in out


def test_format_results_says_nothing_extra_for_an_uncapped_record():
    record = rec(1, "ordinary content")
    out = format_results([record])
    assert "cannot be recovered" not in out


# == Phase 2 (task-1271): compute_stats / format_stats (run_log_stats) ======


def test_compute_stats_groups_by_tool_by_default():
    groups, total_matched, omitted = compute_stats(CORPUS)
    by_key = {g.key: g for g in groups}
    # CORPUS: read_file(ok), web_search(error), model turn (tool=""), write_file(ok).
    assert by_key["read_file"].count == 1
    assert by_key["read_file"].error_count == 0
    assert by_key["web_search"].count == 1
    assert by_key["web_search"].error_count == 1
    assert by_key["write_file"].count == 1
    assert total_matched == len(CORPUS)
    assert omitted == 0


def test_compute_stats_unrecognised_group_by_falls_back_to_tool():
    default_groups, default_total, default_omitted = compute_stats(CORPUS, group_by="tool")
    junk_groups, junk_total, junk_omitted = compute_stats(CORPUS, group_by="not_a_real_field")
    assert junk_groups == default_groups
    assert junk_total == default_total
    assert junk_omitted == default_omitted


def test_compute_stats_group_by_status_and_type():
    by_status = {g.key: g.count for g in compute_stats(CORPUS, group_by="status")[0]}
    assert by_status.get("ok") == 2
    assert by_status.get("error") == 1
    by_type = {g.key: g.count for g in compute_stats(CORPUS, group_by="type")[0]}
    assert by_type.get("model") == 1
    assert by_type.get("tool_result") == 3


def test_compute_stats_pre_filters_compose_before_grouping():
    groups, total_matched, _omitted = compute_stats(CORPUS, group_by="tool", status="ok")
    assert {g.key for g in groups} == {"read_file", "write_file"}
    assert total_matched == 2


def test_compute_stats_from_to_record_bounds_compose_with_group_by():
    groups, total_matched, _omitted = compute_stats(
        CORPUS, group_by="tool", from_record=2, to_record=3
    )
    # Only records 2 (web_search) and 3 (model, tool="") fall in range.
    assert {g.key for g in groups} == {"web_search", "-"}
    assert total_matched == 2


def test_compute_stats_content_bytes_sums_utf8_length_not_char_count():
    # A multi-byte character makes UTF-8 byte length diverge from len().
    single = [rec(1, "héllo", tool="x")]
    groups, _total, _omitted = compute_stats(single, group_by="tool")
    assert groups[0].content_bytes == len("héllo".encode("utf-8"))
    assert groups[0].content_bytes != len("héllo")


def test_compute_stats_sorted_by_descending_count_then_key():
    many = [rec(i, "x", tool="a") for i in range(1, 4)] + [rec(4, "x", tool="b")]
    groups, _total, _omitted = compute_stats(many, group_by="tool")
    assert [g.key for g in groups] == ["a", "b"]
    assert groups[0].count == 3 and groups[1].count == 1


def test_compute_stats_empty_log_returns_no_groups():
    assert compute_stats([]) == ([], 0, 0)
    groups, total, omitted = compute_stats([])
    assert format_stats(groups, group_by="tool", total_records=total, omitted_groups=omitted) == (
        "No records matched."
    )


def test_compute_stats_output_is_bounded_by_distinct_groups_not_record_count():
    # A log "large enough that an unbounded implementation would blow up":
    # thousands of records collapsing into a handful of tool names -- well
    # under MAX_STATS_GROUPS, so nothing here is expected to be capped.
    tools = ["alpha", "beta", "gamma", "delta", "epsilon"]
    huge = [
        rec(
            i,
            f"record body {i}",
            tool=tools[i % len(tools)],
            status="ok" if i % 7 else "error",
        )
        for i in range(1, 5001)
    ]
    groups, total_matched, omitted = compute_stats(huge, group_by="tool")
    assert len(groups) == len(tools)  # bounded by distinct tools, not 5000
    assert total_matched == 5000
    assert omitted == 0
    rendered = format_stats(groups, group_by="tool", total_records=total_matched)
    assert rendered.count("\n") == len(tools)  # one header + one line/group
    assert len(rendered) < 2000  # nowhere near what a 5000-record dump would be


# -- A (Qodo review, PR #1078): group cap must be enforced, not just the -----
# -- record-vs-group distinction above ---------------------------------------
#
# `compute_stats`' own boundedness claim ("output scales with distinct
# GROUPS, never records") is false for `group_by="tool"` unless the number
# of distinct groups is ITSELF capped -- tool names come from the model or
# an MCP server, a set this module does not control. This drives many more
# distinct tool names through `compute_stats` than `MAX_STATS_GROUPS`
# allows, and confirms: (a) the returned group list is capped at
# `MAX_STATS_GROUPS`; (b) the highest-count groups survive, never an
# arbitrary subset; (c) the omitted count is exact; (d) `format_stats`
# reports the omission explicitly rather than silently rendering a partial
# list as if it were the whole picture.


def test_compute_stats_caps_group_count_when_distinct_tool_names_exceed_the_limit():
    # MAX_STATS_GROUPS + 37 distinct tool names, each called a DIFFERENT
    # number of times so "highest count survives" is unambiguous to assert:
    # tool_0 called (n) times, tool_1 called (n-1) times, ... -- strictly
    # descending, no ties to worry about.
    distinct_tool_count = MAX_STATS_GROUPS + 37
    records = []
    number = 1
    for i in range(distinct_tool_count):
        calls = distinct_tool_count - i  # tool_0 most frequent, last least
        for _ in range(calls):
            records.append(rec(number, "x", tool=f"tool_{i}"))
            number += 1

    groups, total_matched, omitted = compute_stats(records, group_by="tool")

    assert total_matched == len(records)  # every record still counted
    assert len(groups) == MAX_STATS_GROUPS  # rendered list IS capped
    assert omitted == distinct_tool_count - MAX_STATS_GROUPS  # exact remainder
    # The surviving groups are exactly the MAX_STATS_GROUPS highest-count
    # ones (tool_0 .. tool_{MAX_STATS_GROUPS - 1}), in descending order.
    assert [g.key for g in groups] == [f"tool_{i}" for i in range(MAX_STATS_GROUPS)]
    assert all(
        groups[i].count >= groups[i + 1].count for i in range(len(groups) - 1)
    )

    rendered = format_stats(
        groups, group_by="tool", total_records=total_matched, omitted_groups=omitted
    )
    # Rendered output stays bounded: 1 header line + MAX_STATS_GROUPS group
    # lines + 1 trailer line = MAX_STATS_GROUPS + 2 lines (MAX_STATS_GROUPS
    # + 1 newlines) -- never one line per distinct tool name (that would be
    # distinct_tool_count + 1 lines), and the omission is stated, not
    # silently cut.
    assert rendered.count("\n") == MAX_STATS_GROUPS + 1
    assert f"{omitted}" in rendered
    assert "omitted" in rendered


def test_compute_stats_zero_max_groups_still_reports_the_exact_total_matched():
    # max_groups=0 is an edge case (not exercised by the real closure, which
    # always uses the module default) -- confirm the cap logic degrades
    # sanely rather than treating 0 as "no cap" by accident.
    records = [rec(i, "x", tool=f"tool_{i}") for i in range(1, 6)]
    groups, total_matched, omitted = compute_stats(records, group_by="tool", max_groups=0)
    assert total_matched == 5
    # max_groups=0 is falsy -> the `max_groups > 0` guard treats it as
    # "no cap requested" (consistent with from_record/to_record's own
    # falsy-means-unset convention elsewhere in this module) rather than
    # "cap at zero groups".
    assert len(groups) == 5
    assert omitted == 0


# == Phase 2 (task-1271): slice_records / format_slice (run_log_slice) ======


def test_slice_records_selects_the_requested_range():
    log = [rec(i, f"body {i}") for i in range(1, 11)]
    selected, total, lo, hi = slice_records(log, from_record=3, to_record=5)
    assert [r.number for r in selected] == [3, 4, 5]
    assert (total, lo, hi) == (3, 3, 5)


def test_slice_records_defaults_to_a_fixed_width_window():
    log = [rec(i, f"body {i}") for i in range(1, 100)]
    selected, total, lo, hi = slice_records(log, from_record=10)
    assert lo == 10 and hi == 10 + DEFAULT_SLICE_WIDTH - 1
    assert [r.number for r in selected] == list(range(10, hi + 1))


def test_slice_records_clamps_a_below_range_from_record_to_1():
    log = [rec(i, f"body {i}") for i in range(1, 5)]
    selected, total, lo, hi = slice_records(log, from_record=-99, to_record=2)
    assert lo == 1
    assert [r.number for r in selected] == [1, 2]


def test_slice_records_to_below_from_collapses_to_one_record():
    log = [rec(i, f"body {i}") for i in range(1, 5)]
    selected, total, lo, hi = slice_records(log, from_record=3, to_record=1)
    assert (lo, hi) == (3, 3)
    assert [r.number for r in selected] == [3]


def test_slice_records_empty_log_returns_nothing_but_still_resolves_bounds():
    selected, total, lo, hi = slice_records([], from_record=5, to_record=9)
    assert selected == []
    assert (total, lo, hi) == (0, 5, 9)


def test_slice_records_output_is_bounded_regardless_of_requested_width_or_log_size():
    # A log large enough, and a range wide enough, that an unbounded
    # implementation would return everything.
    huge = [rec(i, f"body {i}") for i in range(1, 10_001)]
    selected, total, lo, hi = slice_records(huge, from_record=1, to_record=10_000)
    assert len(selected) == MAX_SLICE_RECORDS
    assert total == 10_000  # what WOULD have matched, before the cap
    assert [r.number for r in selected] == list(range(1, MAX_SLICE_RECORDS + 1))


def test_format_slice_reports_the_range_and_reuses_format_results_rendering():
    log = [rec(i, f"content {i}") for i in range(1, 4)]
    selected, total, lo, hi = slice_records(log, from_record=1, to_record=3)
    rendered = format_slice(selected, from_record=lo, to_record=hi, total_matched=total)
    # format_results' own "record NNNNNN [...]" header style is present --
    # confirming reuse, not a second, divergent renderer.
    assert "record 000001" in rendered and "record 000003" in rendered
    assert "content 1" in rendered and "content 3" in rendered


def test_format_slice_notes_clipping_and_the_next_from_record():
    huge = [rec(i, f"body {i}") for i in range(1, 200)]
    selected, total, lo, hi = slice_records(huge, from_record=1, to_record=100)
    rendered = format_slice(selected, from_record=lo, to_record=hi, total_matched=total)
    assert f"showing {len(selected)} of {total}" in rendered
    assert f"from_record={selected[-1].number + 1}" in rendered


def test_format_slice_no_clipping_note_when_everything_fit():
    log = [rec(i, f"body {i}") for i in range(1, 4)]
    selected, total, lo, hi = slice_records(log, from_record=1, to_record=3)
    rendered = format_slice(selected, from_record=lo, to_record=hi, total_matched=total)
    assert "continue with from_record" not in rendered


def test_format_slice_empty_range_reads_distinctly_from_no_search_hits():
    rendered = format_slice([], from_record=500, to_record=510, total_matched=0)
    assert rendered == "No records numbered 000500-000510 in this run's log."
    # Distinct wording from format_results' own empty message, so a model
    # can tell "nothing in this range" apart from "no query hits".
    assert rendered != format_results([])
