"""Real-process bounds for custom trace PII regular expressions."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

from tldw_chatbook.Chat.console_trace_custom_pii import CustomPIIRule
from tldw_chatbook.Chat.console_trace_regex_worker import (
    CUSTOM_PII_WORKER_CRASH,
    CUSTOM_PII_WORKER_INPUT_LIMIT,
    CUSTOM_PII_WORKER_INVALID_BATCH,
    CUSTOM_PII_WORKER_MALFORMED_OUTPUT,
    CUSTOM_PII_WORKER_MATCH_LIMIT,
    CUSTOM_PII_WORKER_OUTPUT_LIMIT,
    CUSTOM_PII_WORKER_TIMEOUT,
    CustomPIIWorkerLimits,
    _worker_match,
    run_custom_pii_batch,
)


def _rule(
    pattern: str,
    *,
    rule_id: str = "customer-id",
    category: str = "customer_id",
) -> CustomPIIRule:
    return CustomPIIRule(
        rule_id=rule_id,
        label="Customer ID",
        category=category,
        pattern=pattern,
        priority=10,
    )


def test_worker_returns_only_content_free_unicode_spans() -> None:
    source = {"account": "👩🏽‍💻 customer-12345678"}

    result = run_custom_pii_batch(source, (_rule(r"customer-\d{8}"),))

    assert result.available is True
    assert result.omission_reason_code is None
    assert result.worker_terminated is False
    assert [
        (
            item.field_path,
            item.span.start_codepoint,
            item.span.end_codepoint,
            item.span.category,
            item.span.rule_id,
        )
        for item in result.field_redactions
    ] == [("$/@0", 5, 22, "customer_id", "customer-id")]
    assert "customer-12345678" not in repr(result)
    if sys.platform.startswith("linux"):
        assert "memory" in result.enforced_limits
    if os.name == "posix":
        assert "cpu" in result.enforced_limits
        assert "output" in result.enforced_limits


def test_parent_input_limit_fails_before_worker_execution() -> None:
    result = run_custom_pii_batch(
        {"value": "private-value"},
        (_rule("private"),),
        limits=CustomPIIWorkerLimits(max_input_bytes=32),
    )

    assert result.available is False
    assert result.omission_reason_code == CUSTOM_PII_WORKER_INPUT_LIMIT
    assert result.worker_terminated is False


def test_parent_rejects_unvalidated_rule_objects_before_serializing() -> None:
    with pytest.raises(TypeError, match="rules"):
        run_custom_pii_batch({"value": "private-value"}, (object(),))  # type: ignore[arg-type]


def test_dense_matches_and_output_overflow_fail_closed() -> None:
    dense = run_custom_pii_batch(
        {"value": "a" * 20},
        (_rule("a"),),
        limits=CustomPIIWorkerLimits(max_matches=3),
    )
    output = run_custom_pii_batch(
        {"value": "a" * 20},
        (_rule("a"),),
        limits=CustomPIIWorkerLimits(max_output_bytes=128),
    )

    assert dense.available is False
    assert dense.omission_reason_code == CUSTOM_PII_WORKER_MATCH_LIMIT
    assert output.available is False
    assert output.omission_reason_code == CUSTOM_PII_WORKER_OUTPUT_LIMIT


@pytest.mark.parametrize(
    ("script", "reason"),
    [
        ("raise SystemExit(7)\n", CUSTOM_PII_WORKER_CRASH),
        ("print('not-json')\n", CUSTOM_PII_WORKER_MALFORMED_OUTPUT),
        (
            'print(\'{"version":1,"outcome":"applied","matches":[],'
            '"enforced_limits":[],"unexpected":true}\')\n',
            CUSTOM_PII_WORKER_MALFORMED_OUTPUT,
        ),
        (
            'print(\'{"version":true,"outcome":"applied",'
            '"matches":[],"enforced_limits":[]}\')\n',
            CUSTOM_PII_WORKER_MALFORMED_OUTPUT,
        ),
        (
            'print(\'{"version":1,"version":1,"outcome":"applied",'
            '"matches":[],"enforced_limits":[]}\')\n',
            CUSTOM_PII_WORKER_MALFORMED_OUTPUT,
        ),
        (
            'print(\'{"version":1,"outcome":"applied","matches":[{'
            '"field_path":3,"start_codepoint":0,"end_codepoint":1,'
            '"category":"custom","rule_id":"rule"}],'
            '"enforced_limits":[]}\')\n',
            CUSTOM_PII_WORKER_MALFORMED_OUTPUT,
        ),
    ],
)
def test_crashing_or_malformed_worker_fails_content_free(
    tmp_path: Path,
    script: str,
    reason: str,
) -> None:
    worker = tmp_path / "worker.py"
    worker.write_text(script, encoding="utf-8")

    result = run_custom_pii_batch(
        {"value": "private-value"},
        (_rule("private"),),
        worker_path=worker,
    )

    assert result.available is False
    assert result.omission_reason_code == reason
    assert "private-value" not in repr(result)


def test_child_rejects_boolean_or_extra_limit_fields() -> None:
    base = {
        "version": 1,
        "value": {"value": "private-value"},
        "rules": [],
        "limits": {
            "max_fields": 10,
            "max_field_codepoints": 100,
            "max_rules": 10,
            "max_matches": 10,
        },
    }
    boolean = {**base, "limits": {**base["limits"], "max_fields": True}}
    extra = {**base, "limits": {**base["limits"], "extra": 1}}

    for request in (boolean, extra):
        result = _worker_match(request, ())
        assert result["outcome"] == "omitted"
        assert result["reason"] == CUSTOM_PII_WORKER_INVALID_BATCH


def test_catastrophic_backtracking_is_killed_by_batch_deadline() -> None:
    result = run_custom_pii_batch(
        {"value": "a" * 30_000 + "!"},
        (_rule(r"(a+)+$"),),
        limits=CustomPIIWorkerLimits(deadline_ms=50),
    )

    assert result.available is False
    assert result.omission_reason_code == CUSTOM_PII_WORKER_TIMEOUT
    assert result.worker_terminated is True
