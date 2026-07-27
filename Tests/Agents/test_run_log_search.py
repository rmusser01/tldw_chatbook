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
