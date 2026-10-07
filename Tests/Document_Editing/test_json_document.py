"""Real strict JSON/JSONL grammar and source-token formatting controls."""

import json
import re
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

CASES = json.loads(
    (Path(__file__).parents[1] / "fixtures/document_editing/v1/cases.json").read_text()
)


def test_format_keeps_large_number_and_is_idempotent(request_for):
    from tldw_chatbook.Document_Editing.engine import format_document

    first = format_document(request_for('{"n":9007199254740993,"x":-0}'))
    assert first.state == "formatted"
    assert "9007199254740993" in first.replacement
    assert '"x": -0' in first.replacement
    second = format_document(request_for(first.replacement))
    assert second.state == "unchanged"
    assert second.replacement is None


@pytest.mark.parametrize(
    "case", [c for c in CASES if c["language"] != "yaml"], ids=lambda c: c["id"]
)
def test_shared_json_cases(request_for, case):
    from tldw_chatbook.Document_Editing.engine import check_document, format_document

    req = request_for(case["text"], language=case["language"])
    check = check_document(req)
    assert check.state == "checked" and check.complete
    assert (not any(d.severity == "error" for d in check.diagnostics)) == case["valid"]
    assert {d.code for d in check.diagnostics} == set(case["codes"])
    assert check.format_allowed == (case["format"] == "allowed")
    result = format_document(req)
    assert result.key == req.key
    if case["format"] == "blocked":
        assert result.state == "blocked" and result.replacement is None
    else:
        assert result.state in {"formatted", "unchanged"}
        output = result.replacement if result.replacement is not None else req.text
        assert all(token in output for token in case["must_keep"])
        assert (
            format_document(request_for(output, language=req.language)).state
            == "unchanged"
        )


@pytest.mark.parametrize(
    "source",
    ["[1,]", "01", "+1", "1.", '"bad\nstring"', "\ufeff{}", "undefined", "{} {}"],
)
def test_strict_rejections(request_for, source):
    from tldw_chatbook.Document_Editing.engine import format_document

    result = format_document(request_for(source))
    assert result.state == "blocked"
    assert result.check.diagnostics[0].code == "syntax"


def test_duplicate_span_and_scope(request_for):
    from tldw_chatbook.Document_Editing.engine import check_document

    source = '{"😀":0,"é":1,"\\u0065́":2,"x":{"é":3}}'
    result = check_document(request_for(source))
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert source[diagnostic.start : diagnostic.end] == '"\\u0065́"'
    assert diagnostic.severity == "warning"


def test_jsonl_records_newlines_and_bom_metadata(request_for):
    from tldw_chatbook.Document_Editing.engine import check_document, format_document

    source = ' { "n" : 1e400 } \r\n [ -0, "\\u0061" ] '
    result = format_document(request_for(source, language="jsonl", newline="\r\n"))
    assert result.replacement == '{"n":1e400}\r\n[-0,"\\u0061"]'
    bad = check_document(request_for("{}\n", language="jsonl", has_bom=True))
    assert not bad.format_allowed and bad.diagnostics[0].code == "syntax"
    assert bad.diagnostics[0].start == 0
    good = check_document(request_for('"a\u2028b"\n', language="jsonl"))
    assert good.format_allowed


def test_jsonl_offsets_and_diagnostic_limit(request_for):
    from tldw_chatbook.Document_Editing.engine import check_document

    source = '"😀"\r\n{\r\n'
    result = check_document(request_for(source, language="jsonl"))
    assert result.diagnostics[0].start == 6
    limited = check_document(request_for("\n" * 101, language="jsonl"))
    assert len(limited.diagnostics) == 100 and limited.truncated
    assert not limited.complete and not limited.format_allowed


@pytest.mark.parametrize("language", [None, "text", "markdown", "future"])
def test_unsupported_modes_never_parse(request_for, language):
    from tldw_chatbook.Document_Editing.engine import check_document

    result = check_document(request_for("{", language=language))
    assert result.state == "unsupported" and not result.complete


def test_excerpt_size_and_depth_bounds(request_for):
    from tldw_chatbook.Document_Editing.engine import check_document

    assert check_document(request_for("{}", is_excerpt=True)).state == "unsupported"
    assert (
        check_document(request_for('"' + "😀" * 262144 + '"')).state
        == "resource_limited"
    )
    assert (
        check_document(request_for("[" * 101 + "0" + "]" * 101)).state
        == "resource_limited"
    )
    assert check_document(request_for("[" * 100 + "0" + "]" * 100)).format_allowed
    assert check_document(request_for("9" * 5000)).format_allowed


def test_normalized_errors_never_echo_source(request_for):
    from tldw_chatbook.Document_Editing.engine import check_document

    result = check_document(request_for('{"PRIVATE_SENTINEL":SECRET}'))
    assert (
        result.diagnostics
        and "PRIVATE" not in repr(result)
        and "SECRET" not in repr(result)
    )


JSON_VALUES = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(),
    lambda child: st.lists(child, max_size=5)
    | st.dictionaries(st.text(), child, max_size=5),
    max_leaves=30,
)


@given(JSON_VALUES)
@settings(
    max_examples=120,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_generated_json_tokens_and_idempotence(request_for, value):
    from tldw_chatbook.Document_Editing.engine import format_document

    source = json.dumps(value, ensure_ascii=True, separators=(" , ", " : "))
    result = format_document(request_for(source))
    assert result.state in {"formatted", "unchanged"}
    output = result.replacement if result.replacement is not None else source
    pattern = r'"(?:\\.|[^"\\])*"|[^\s]'
    assert re.findall(pattern, output) == re.findall(pattern, source)
    assert json.loads(output) == value
    assert format_document(request_for(output)).state == "unchanged"


def test_cooperative_deadline_is_a_resource_result(request_for, monkeypatch):
    from tldw_chatbook.Document_Editing import source_positions
    from tldw_chatbook.Document_Editing.engine import format_document

    ticks = iter([0.0, 3.0])
    monkeypatch.setattr(source_positions, "monotonic", lambda: next(ticks))
    result = format_document(request_for("{}"))
    assert result.state == "resource_limited"
    assert not result.check.complete and result.replacement is None


def test_unexpected_failure_is_not_a_resource_success(request_for, monkeypatch):
    from tldw_chatbook.Document_Editing import json_document
    from tldw_chatbook.Document_Editing.engine import check_document

    def broken(*_args):
        raise RuntimeError("PRIVATE_SENTINEL")

    monkeypatch.setattr(json_document, "tokens", broken)
    result = check_document(request_for("{}"))
    assert result.state == "failed" and not result.complete
    assert "PRIVATE_SENTINEL" not in repr(result)
