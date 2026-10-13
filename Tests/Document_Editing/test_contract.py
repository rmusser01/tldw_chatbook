"""Pure contracts only; language engines consume these cases in later tasks."""

import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from tldw_chatbook.Document_Editing.models import CheckResult, Diagnostic, FormatResult


def checked(request, **overrides):
    fields = {
        "state": "checked",
        "complete": True,
        "diagnostics": (),
        "truncated": False,
        "format_allowed": True,
    }
    fields.update(overrides)
    return CheckResult.for_request(request, **fields)


@pytest.mark.parametrize("start,end", [(-1, 0), (2, 1)])
def test_invalid_ranges(start, end):
    with pytest.raises(ValueError):
        Diagnostic("syntax", "error", "Invalid", start, end)


def test_request_identity_and_future_mode(request_for):
    first = request_for("x", language="future", draft_revision=2, newline="\r\n")
    second = request_for("x")
    assert first.key.request_id != second.key.request_id
    assert first.language == "future" and first.key.draft_revision == 2
    with pytest.raises(FrozenInstanceError):
        first.key.document_id = "changed"


@pytest.mark.parametrize(
    "fields",
    [
        {"complete": False},
        {"truncated": True},
        {"state": "unsupported"},
        {"diagnostics": (Diagnostic("syntax", "error", "Invalid", 0, 1),)},
        {"diagnostics": (Diagnostic("duplicate_key", "warning", "Duplicate", 0, 1),)},
    ],
)
def test_false_format_admission(request_for, fields):
    with pytest.raises(ValueError):
        checked(request_for("{}"), **fields)


@pytest.mark.parametrize("state", ["unsupported", "resource_limited", "failed"])
def test_nonchecks_cannot_be_complete(request_for, state):
    with pytest.raises(ValueError):
        checked(request_for("{}"), state=state, format_allowed=False)


def test_excerpt_cannot_report_success(request_for):
    with pytest.raises(ValueError):
        checked(request_for("{}", is_excerpt=True), format_allowed=False)


def test_request_ranges_bounded_by_unicode_scalars(request_for):
    with pytest.raises(ValueError):
        checked(
            request_for("😀"),
            diagnostics=(Diagnostic("syntax", "error", "Invalid", 0, 2),),
            format_allowed=False,
        )


@pytest.mark.parametrize(
    "state", ["blocked", "unchanged", "resource_limited", "failed"]
)
def test_replacement_only_on_formatted(request_for, state):
    req = request_for("{}")
    with pytest.raises(ValueError):
        FormatResult(req.key, state, "{ }", checked(req))


def test_success_requires_authoritative_check_and_matching_key(request_for):
    req = request_for("{}")
    blocked = checked(req, format_allowed=False)
    with pytest.raises(ValueError):
        FormatResult(req.key, "formatted", "{ }", blocked)
    with pytest.raises(ValueError):
        FormatResult(request_for("{}").key, "formatted", "{ }", checked(req))
    with pytest.raises(ValueError):
        FormatResult(req.key, "formatted", None, checked(req))
    assert FormatResult(req.key, "formatted", "", checked(req)).replacement == ""
    assert FormatResult(req.key, "unchanged", None, checked(req)).state == "unchanged"


def test_fixture_v1_schema_and_source():
    path = Path(__file__).parents[1] / "fixtures/document_editing/v1/cases.json"
    raw = path.read_bytes()
    cases = json.loads(raw)
    assert (
        hashlib.sha256(raw).hexdigest()
        == "fcf516e4c0bb8ff7a19748a4d5f7deb8a30ad0078ea0ce2af0e1e81af50fb0d7"
    )
    assert len({case["id"] for case in cases}) == len(cases)
    for case in cases:
        assert set(case) == {
            "id",
            "language",
            "text",
            "valid",
            "codes",
            "format",
            "must_keep",
        }
        assert case["language"] in {"json", "jsonl", "yaml"}
        assert isinstance(case["id"], str) and case["id"]
        assert isinstance(case["text"], str)
        assert isinstance(case["codes"], list) and all(
            isinstance(code, str) for code in case["codes"]
        )
        assert isinstance(case["must_keep"], list) and all(
            isinstance(token, str) for token in case["must_keep"]
        )
        assert type(case["valid"]) is bool
        assert case["format"] in {"allowed", "blocked"}
        assert all(token in case["text"] for token in case["must_keep"])
        if "unsupported" in case["codes"]:
            assert (
                case["valid"]
                and "syntax" not in case["codes"]
                and case["format"] == "blocked"
            )
        if not case["valid"] or case["codes"]:
            assert case["format"] == "blocked"
    by_id = {case["id"]: case for case in cases}
    assert by_id["json-escaped-duplicate"]["text"] == r'{"a":1,"\u0061":2}'
    assert by_id["jsonl-blank"]["text"] == "{}\n\n{}\n"
    assert by_id["jsonl-empty"]["text"] == ""
    print("fixture v1 SHA-256:", hashlib.sha256(raw).hexdigest())
