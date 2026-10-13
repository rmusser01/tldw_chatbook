"""Qualify actual YAML source edits and symbolic graph preservation."""

import json
from pathlib import Path

import pytest

CASES = json.loads(
    (Path(__file__).parents[1] / "fixtures/document_editing/v1/cases.json").read_text()
)


@pytest.mark.parametrize(
    "case", [c for c in CASES if c["language"] == "yaml"], ids=lambda c: c["id"]
)
def test_shared_yaml_cases(request_for, case):
    from tldw_chatbook.Document_Editing.engine import check_document, format_document

    req = request_for(case["text"], language="yaml")
    check = check_document(req)
    assert {d.code for d in check.diagnostics} == set(case["codes"])
    if "unsupported" in case["codes"]:
        assert check.state == "unsupported" and not check.complete
    else:
        assert check.state == "checked" and check.complete
        assert (not any(d.severity == "error" for d in check.diagnostics)) == case[
            "valid"
        ]
    assert check.format_allowed == (case["format"] == "allowed")
    result = format_document(req)
    if case["format"] == "blocked":
        assert result.state == "blocked" and result.replacement is None
    else:
        assert result.state in {"formatted", "unchanged"}
        output = result.replacement if result.replacement is not None else req.text
        assert all(token in output for token in case["must_keep"])
        assert (
            format_document(request_for(output, language="yaml")).state == "unchanged"
        )


def test_nontrivial_stream_format_preserves_tokens_comments_graph(request_for):
    from tldw_chatbook.Document_Editing.engine import format_document

    source = (
        "%YAML 1.2\n---\n# header\nbase: &base\n    n:   9007199254740993 # big\n"
        '    x: -0\ncopy:\n    <<: *base\n    quoted: "\\u0061"\n...\n---\n'
        "block: |-\n  first\n  second\nflow: {a:   1e400, b: 0.1234567890123456789}\n"
    )
    result = format_document(request_for(source, language="yaml"))
    assert result.state == "formatted"
    expected = (
        source.replace("    n:   ", "  n: ")
        .replace("    x:", "  x:")
        .replace("    <<:", "  <<:")
        .replace("    quoted:", "  quoted:")
        .replace("a:   ", "a: ")
    )
    assert result.replacement == expected
    assert format_document(request_for(expected, language="yaml")).state == "unchanged"


@pytest.mark.parametrize(
    "source",
    [
        'a: 1\n"a": 2\n',
        "0xB: x\n11: y\n",
        "%YAML 1.1\n---\nyes: x\ntrue: y\n",
        "%YAML 1.1\n---\n013: x\n11: y\n",
        "1.0: x\n1e0: y\n",
        "null: x\n~: y\n",
    ],
)
def test_version_aware_duplicate_scalars(request_for, source):
    from tldw_chatbook.Document_Editing.engine import check_document

    result = check_document(request_for(source, language="yaml"))
    assert [d.code for d in result.diagnostics] == ["duplicate_key"]
    assert result.diagnostics[0].severity == "error"


def test_core_keys_are_distinct_and_merge_override_is_not_duplicate(request_for):
    from tldw_chatbook.Document_Editing.engine import check_document

    source = "yes: a\ntrue: b\n1: c\n1.0: d\nbase: &b {x: 1}\ncopy: {<<: *b, x: 2}\n"
    assert check_document(request_for(source, language="yaml")).format_allowed


@pytest.mark.parametrize(
    "source",
    [
        "value: !custom tagged\n",
        "? [a, b]\n: value\n",
        "!!python/object/apply:os.system [echo SENTINEL]\n",
        "%YAML 1.3\n---\na: 1\n",
    ],
)
def test_unqualified_never_constructs_or_formats(request_for, source):
    from tldw_chatbook.Document_Editing.engine import format_document

    result = format_document(request_for(source, language="yaml"))
    assert result.state == "blocked" and result.check.state == "unsupported"
    assert result.check.diagnostics[0].code == "unsupported"


def test_depth_and_alias_graph_without_expansion(request_for):
    from tldw_chatbook.Document_Editing.engine import check_document, format_document

    assert (
        check_document(request_for("[" * 101 + "0" + "]" * 101, language="yaml")).state
        == "resource_limited"
    )
    assert check_document(
        request_for("[" * 100 + "0" + "]" * 100, language="yaml")
    ).format_allowed
    source = "a: &a [one, two]\nb: &b [*a, *a, *a]\nc: [*b, *b, *b]\n"
    result = format_document(request_for(source, language="yaml"))
    assert result.state in {"formatted", "unchanged"}
    assert "*b" in (result.replacement or source)
    undefined = check_document(request_for("a: *missing\n", language="yaml"))
    assert undefined.diagnostics[0].code == "syntax"


def test_yaml_safe_error_and_newline_preservation(request_for):
    from tldw_chatbook.Document_Editing.engine import check_document, format_document

    result = check_document(request_for("PRIVATE_SENTINEL: [", language="yaml"))
    assert result.diagnostics and "PRIVATE_SENTINEL" not in repr(result)
    source = "root:\r\n    a:   1"
    result = format_document(request_for(source, language="yaml", newline="\r\n"))
    assert result.replacement == "root:\r\n  a: 1"


@pytest.mark.parametrize(
    "source",
    [
        "a: !!bool nonsense\n",
        "!!str {a: 1}\n",
        "!!seq {a: 1}\n",
        "!!int nope: value\n",
        'a: !!int "09"\n%YAML 1.1\n',
    ],
)
def test_unqualified_explicit_tags_do_not_invent_values(request_for, source):
    from tldw_chatbook.Document_Editing.engine import check_document

    result = check_document(request_for(source, language="yaml"))
    assert not result.format_allowed
    assert result.state in {"unsupported", "checked"}
    assert result.diagnostics


@pytest.mark.parametrize(
    "source", ["a: {<<: 12}\n", "a: {<<: [1, 2]}\n", "base: &base 12\na: {<<: *base}\n"]
)
def test_unqualified_merge_targets_are_not_claimed_valid(request_for, source):
    from tldw_chatbook.Document_Editing.engine import check_document

    result = check_document(request_for(source, language="yaml"))
    assert result.state == "unsupported" and not result.complete


@pytest.mark.parametrize("replacement", ["2", "[", "1 # LOST"])
def test_failed_formatter_invariant_retains_draft(
    request_for, monkeypatch, replacement
):
    from tldw_chatbook.Document_Editing import yaml_document
    from tldw_chatbook.Document_Editing.engine import format_document

    monkeypatch.setattr(yaml_document, "_edits", lambda *_: [(3, 4, replacement)])
    result = format_document(request_for("a: 1\n", language="yaml"))
    assert result.state == "failed" and result.replacement is None
    assert result.check.diagnostics[0].code == "failed"


def test_multiline_scalar_indentation_declines_without_changing_content(request_for):
    from tldw_chatbook.Document_Editing.engine import format_document

    source = "root:\n    text: |\n      keep indentation\n"
    result = format_document(request_for(source, language="yaml"))
    assert result.state == "blocked" and result.check.state == "unsupported"
    assert result.replacement is None


def test_scalar_end_does_not_protect_next_sibling(request_for):
    from tldw_chatbook.Document_Editing.engine import format_document

    source = "root:\n    text: x\nother: |\n  value\nnext:\n    a: 1\n"
    result = format_document(request_for(source, language="yaml"))
    assert result.replacement == "root:\n  text: x\nother: |\n  value\nnext:\n  a: 1\n"


def test_core_timestamp_and_float_precision_key_semantics(request_for):
    from tldw_chatbook.Document_Editing.engine import check_document

    assert (
        check_document(request_for('2026-01-01: x\n"2026-01-01": y\n', language="yaml"))
        .diagnostics[0]
        .code
        == "duplicate_key"
    )
    source = (
        "0.123456789012345678901234567891: a\n0.123456789012345678901234567892: b\n"
    )
    assert check_document(request_for(source, language="yaml")).format_allowed


def test_billion_laughs_is_only_a_symbolic_graph(request_for):
    from tldw_chatbook.Document_Editing.engine import format_document

    source = "a: &a [x, x, x]\n"
    previous = "a"
    for index in range(20):
        name = f"n{index}"
        source += f'{name}: &{name} [{", ".join(["*" + previous] * 10)}]\n'
        previous = name
    result = format_document(request_for(source, language="yaml"))
    assert result.state == "unchanged"


def test_numeric_key_implementation_limit_is_not_syntax(request_for):
    from tldw_chatbook.Document_Editing.engine import check_document

    result = check_document(
        request_for("? " + "9" * 5000 + "\n: value\n", language="yaml")
    )
    assert result.state == "resource_limited"
    assert {d.code for d in result.diagnostics} == {"resource_limited"}
    assert check_document(
        request_for("value: " + "9" * 5000 + "\n", language="yaml")
    ).format_allowed


def test_yaml_diagnostic_display_cap_is_not_complete(request_for):
    from tldw_chatbook.Document_Editing.engine import check_document

    result = check_document(request_for("a: 1\n" * 102, language="yaml"))
    assert len(result.diagnostics) == 100 and result.truncated
    assert not result.complete and not result.format_allowed


def test_nested_sequences_preserve_final_newline_and_are_idempotent(request_for):
    from tldw_chatbook.Document_Editing.engine import format_document

    source = "root:\n    - a\n    - name: n\n      nested:\n          b: 1\n"
    expected = "root:\n  - a\n  - name: n\n    nested:\n      b: 1\n"
    result = format_document(request_for(source, language="yaml"))
    assert result.replacement == expected
    assert format_document(request_for(expected, language="yaml")).state == "unchanged"


def test_indentless_sequence_is_explicitly_unqualified(request_for):
    from tldw_chatbook.Document_Editing.engine import format_document

    result = format_document(request_for("root:\n- a\n- b\n", language="yaml"))
    assert result.check.state == "unsupported" and result.state == "blocked"
    assert result.replacement is None


@pytest.mark.parametrize("directive", ["", "%YAML 1.2\n---\n"])
@pytest.mark.parametrize(
    "spelling, decimal",
    [("+0xB", "11"), ("-0xB", "-11"), ("+0o13", "11"), ("-0o13", "-11")],
)
@pytest.mark.parametrize("quoted", [False, True])
def test_core_signed_base_spellings_are_string_keys(
    request_for, directive, spelling, decimal, quoted
):
    from tldw_chatbook.Document_Editing.engine import format_document

    other = f'"{spelling}"' if quoted else decimal
    source = f"{directive}{spelling}:   a\n{other}: b\n"
    result = format_document(request_for(source, language="yaml"))
    assert result.check.state == "checked" and result.check.complete
    if quoted:
        assert result.state == "blocked" and result.replacement is None
        (diagnostic,) = result.check.diagnostics
        assert diagnostic.code == "duplicate_key" and diagnostic.severity == "error"
        assert source[diagnostic.start : diagnostic.end] == other
    else:
        assert result.check.format_allowed and not result.check.diagnostics
        assert result.state == "formatted"
        assert result.replacement == f"{directive}{spelling}: a\n{other}: b\n"
        assert (
            format_document(request_for(result.replacement, language="yaml")).state
            == "unchanged"
        )


@pytest.mark.parametrize(
    "spelling, decimal",
    [("+0xB", "11"), ("-0xB", "-11"), ("+013", "11"), ("-013", "-11")],
)
@pytest.mark.parametrize("quoted", [False, True])
def test_explicit_yaml11_signed_hex_and_legacy_octal_stay_numeric(
    request_for, spelling, decimal, quoted
):
    from tldw_chatbook.Document_Editing.engine import format_document

    other = f'"{spelling}"' if quoted else decimal
    source = f"%YAML 1.1\n---\n{spelling}:   a\n{other}: b\n"
    result = format_document(request_for(source, language="yaml"))
    if quoted:
        assert result.check.format_allowed and not result.check.diagnostics
        assert result.replacement == f"%YAML 1.1\n---\n{spelling}: a\n{other}: b\n"
    else:
        assert result.state == "blocked" and result.replacement is None
        (diagnostic,) = result.check.diagnostics
        assert diagnostic.code == "duplicate_key" and diagnostic.severity == "error"
