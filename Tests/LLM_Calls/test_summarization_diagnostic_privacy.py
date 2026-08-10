"""Stable, exhaustive guard for summarization diagnostic privacy review."""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
from collections import Counter
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "Tests/fixtures/summarization_diagnostic_review.json"
STARTING_PROJECTION_SHA256 = (
    "a4c9ba5f999199f02fd1c6186d1d88120f6d5f696071127ee192dff2c3503047"
)
MODULE_COUNTS = {
    "tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py": 242,
    "tldw_chatbook/LLM_Calls/Summarization_General_Lib.py": 281,
}
PRIVATE_GROUP_COUNTS = {
    "local_core": 24,
    "local_adapters": 23,
    "local_vllm_ollama": 22,
    "local_custom": 31,
    "general_core": 36,
    "general_mid": 23,
    "general_streaming": 20,
    "general_tail": 20,
}


def _guard() -> ModuleType:
    try:
        return importlib.import_module("Tests.LLM_Calls.summarization_diagnostic_guard")
    except ModuleNotFoundError:
        pytest.fail("summarization diagnostic guard is not implemented")


def _message_shape(expression: str) -> str:
    node = ast.parse(f"logger.info({expression})").body[0]
    assert isinstance(node, ast.Expr)
    assert isinstance(node.value, ast.Call)
    return ast.dump(node.value.args[0], include_attributes=False)


def _single_call(source: str):
    calls = _guard().discover_diagnostic_calls(source, module="synthetic.py")
    assert len(calls) == 1
    return calls[0]


def _ledger_sites() -> list[dict[str, object]]:
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    assert ledger["schema_version"] == 1
    return ledger["sites"]


def _starting_projection(
    sites: list[dict[str, object]],
) -> list[dict[str, object]]:
    projection = []
    for site in sorted(sites, key=lambda item: item["site_id"]):
        record = {
            key: site[key]
            for key in (
                "site_id",
                "module",
                "qualname",
                "group",
                "starting_classification",
            )
        }
        detail = (
            "category"
            if site["starting_classification"] == "private"
            else "safe_reason"
        )
        record[detail] = site[detail]
        record["starting"] = site["starting"]
        projection.append(record)
    return projection


def _call_from_record(site: dict[str, object], record_name: str):
    record = site[record_name]
    assert isinstance(record, dict)
    return _guard().DiagnosticCall(
        module=site["module"],
        qualname=site["qualname"],
        method=record["method"],
        event=record["event"],
        occurrence=record["occurrence"],
        message_shape=record["message_shape"],
        expressions=tuple(record["expressions"]),
        captures_exception=record["captures_exception"],
    )


def test_guard_finds_stdlib_loguru_nested_and_bound_calls() -> None:
    source = """
import logging
from loguru import logger as loguru_logger

audit_logger = logging.getLogger(__name__)

audit_logger.error("stdlib event", account_id)

def outer():
    logger.info("duplicate label", first)
    logger.info("duplicate label", second)

    def stream_generator():
        loguru_logger.bind(session=session_id).opt(colors=True).warning(
            f"stream chunk {chunk.index}", extra_field
        )

    return stream_generator
"""

    calls = _guard().discover_diagnostic_calls(source, module="synthetic.py")

    assert [(call.qualname, call.method, call.event) for call in calls] == [
        ("<module>", "error", "stdlib event"),
        ("outer", "info", "duplicate label"),
        ("outer", "info", "duplicate label"),
        ("outer.stream_generator", "warning", "stream chunk "),
    ]
    assert [call.occurrence for call in calls] == [1, 1, 2, 1]
    assert calls[3].expressions == (
        "chunk.index",
        "extra_field",
        "session_id",
        "True",
    )


def test_guard_identity_ignores_line_movement() -> None:
    compact = """
def summarize():
    logger.info("summary ready", len(summary))
"""
    moved = """


# unrelated navigation-only movement
def summarize():

    logger.info("summary ready", len(summary))
"""

    before = _single_call(compact)
    after = _single_call(moved)

    assert before.identity == after.identity
    assert before == after


def test_guard_rejects_changed_reviewed_safe_expression() -> None:
    starting = _single_call('logger.info("Retry count: {}", retry_count)\n')
    changed = _single_call('logger.info("Retry count: {}", retry_total)\n')

    assert starting.identity == changed.identity
    with pytest.raises(AssertionError, match="frozen diagnostic changed"):
        _guard().assert_review_outcome(starting, changed, outcome="frozen")


def test_guard_accepts_metadata_replacement_with_new_fixed_event() -> None:
    starting = _single_call('logger.error(f"Request failed: {error}")\n')
    repaired = _single_call(
        'logger.error("Request failed; exception_type=%s", exception_type)\n'
    )

    _guard().assert_review_outcome(starting, repaired, outcome="metadata")


def test_guard_records_and_rejects_bare_name_message() -> None:
    call = _single_call("logger.info(message)\n")

    assert call.event == ""
    assert call.message_shape == _message_shape("message")
    assert call.expressions == ("message",)
    with pytest.raises(AssertionError, match="constant string first argument"):
        _guard().assert_review_outcome(call, call, outcome="metadata")


def test_guard_records_and_rejects_percent_formatted_message() -> None:
    call = _single_call('logger.warning("failed: %s" % error)\n')

    assert call.event == "failed: %s"
    assert call.message_shape == _message_shape('"failed: %s" % error')
    assert call.expressions == ("error",)
    with pytest.raises(AssertionError, match="constant string first argument"):
        _guard().assert_review_outcome(call, call, outcome="metadata")


def test_guard_records_and_rejects_dot_format_message() -> None:
    call = _single_call('logger.error("failed: {}".format(error_detail))\n')

    assert call.event == "failed: {}"
    assert call.message_shape == _message_shape('"failed: {}".format(error_detail)')
    assert call.expressions == ("error_detail",)
    with pytest.raises(AssertionError, match="constant string first argument"):
        _guard().assert_review_outcome(call, call, outcome="metadata")


def test_guard_records_and_rejects_concatenated_message() -> None:
    call = _single_call('logger.debug("result: " + result_text)\n')

    assert call.event == "result: "
    assert call.message_shape == _message_shape('"result: " + result_text')
    assert call.expressions == ("result_text",)
    with pytest.raises(AssertionError, match="constant string first argument"):
        _guard().assert_review_outcome(call, call, outcome="metadata")


def test_guard_rejects_exception_and_traceback_capture() -> None:
    source = """
logger.exception("operation failed")
logger.error("operation failed", exc_info=True)
logger.warning("operation failed", stack_info=True)
logger.opt(exception=error).warning("operation failed")
logger.opt(exception=False).warning("ordinary failure")
logger.error("ordinary failure", exc_info=False, stack_info=None)
"""
    calls = _guard().discover_diagnostic_calls(source, module="synthetic.py")

    assert [call.captures_exception for call in calls] == [
        True,
        True,
        True,
        True,
        False,
        False,
    ]
    _guard().assert_review_outcome(calls[0], calls[0], outcome="pending")
    for outcome in ("frozen", "metadata"):
        with pytest.raises(
            AssertionError, match="must not capture exception or traceback"
        ):
            _guard().assert_review_outcome(calls[0], calls[0], outcome=outcome)


def test_ledger_retains_all_523_starting_sites() -> None:
    sites = _ledger_sites()

    assert len(sites) == 523
    assert len({site["site_id"] for site in sites}) == 523
    assert Counter(site["starting_classification"] for site in sites) == {
        "private": 199,
        "reviewed_safe": 324,
    }
    assert Counter(site["module"] for site in sites) == MODULE_COUNTS
    assert (
        Counter(
            site["group"]
            for site in sites
            if site["starting_classification"] == "private"
        )
        == PRIVATE_GROUP_COUNTS
    )

    identities = [_call_from_record(site, "starting").identity for site in sites]
    assert len(set(identities)) == 523
    encoded = json.dumps(
        _starting_projection(sites),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == STARTING_PROJECTION_SHA256


def test_ledger_current_state_matches_sources() -> None:
    sites = _ledger_sites()
    outcomes = {"pending", "frozen", "metadata", "deleted"}
    assert {site["outcome"] for site in sites} <= outcomes

    discovered = []
    for module in MODULE_COUNTS:
        source = (REPO_ROOT / module).read_text(encoding="utf-8")
        discovered.extend(_guard().discover_diagnostic_calls(source, module=module))
    discovered_by_identity = {call.identity: call for call in discovered}
    assert len(discovered_by_identity) == len(discovered), (
        "source contains duplicate diagnostic identities"
    )

    declared_by_identity = {}
    for site in sites:
        starting = _call_from_record(site, "starting")
        if site["outcome"] == "deleted":
            assert site["current"] is None
            assert starting.identity not in discovered_by_identity, (
                f"deleted diagnostic still exists: {site['site_id']}"
            )
            continue

        current = _call_from_record(site, "current")
        assert current.identity not in declared_by_identity, (
            f"duplicate ledger identity: {site['site_id']}"
        )
        declared_by_identity[current.identity] = (site, starting, current)

    assert discovered_by_identity.keys() == declared_by_identity.keys(), (
        "summarization diagnostic calls were added, deleted, or changed "
        "without ledger review"
    )
    for identity, actual in discovered_by_identity.items():
        site, starting, current = declared_by_identity[identity]
        assert actual == current, (
            f"current diagnostic record changed: {site['site_id']}"
        )
        _guard().assert_review_outcome(starting, actual, outcome=site["outcome"])
