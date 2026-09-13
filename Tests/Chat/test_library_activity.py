"""Bounded, content-minimized Console Library activity contracts."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from tldw_chatbook.Chat.library_activity import (
    LIBRARY_ACTIVITY_ERROR_SUMMARY_MAX_CHARS,
    LIBRARY_ACTIVITY_PAYLOAD_MAX_BYTES,
    LIBRARY_ACTIVITY_QUERY_PREVIEW_MAX_CHARS,
    LIBRARY_ACTIVITY_SOURCE_ID_MAX_CHARS,
    LIBRARY_ACTIVITY_SOURCE_REF_MAX_COUNT,
    LIBRARY_ACTIVITY_TITLE_MAX_CHARS,
    LibraryActivityCandidate,
    decode_library_activity_event,
    encode_library_activity_event,
    minimize_library_activity,
)


def _candidate(*, result: object, failure_code: str | None = None):
    return LibraryActivityCandidate(
        attempt_id="attempt-1",
        actor_kind="subagent",
        run_id="run-child",
        parent_run_id="run-parent",
        library_provider="direct",
        operation="library_search_notes",
        arguments={"query": "quarterly plan"},
        structured_result=result,
        failure_code=failure_code,
    )


def test_minimize_activity_keeps_only_bounded_review_fields():
    secret = "sk-" + "a" * 48
    rows = [
        {
            "id": f"note:{index}-" + "x" * 400,
            "type": "note",
            "title": f"Title {index} " + "界" * 300,
            "body": "PRIVATE BODY",
            "snippet": "PRIVATE SNIPPET",
            "excerpt": "PRIVATE EXCERPT",
            "path": "/Users/private/library/note.md",
            "api_key": secret,
        }
        for index in range(12)
    ]
    candidate = _candidate(result={"items": rows, "total": len(rows)})
    candidate = replace(
        candidate,
        arguments={
            "query": "quarterly plan " + secret + " /Users/private/query.txt"
        },
    )

    event = minimize_library_activity(candidate)
    encoded = json.dumps(event.to_payload(), ensure_ascii=False).encode("utf-8")

    assert event.version == 1
    assert event.attempt_id == "attempt-1"
    assert event.run_id == "run-child"
    assert event.actor_kind == "subagent"
    assert event.parent_run_id == "run-parent"
    assert event.status == "succeeded"
    assert event.result_count == 12
    assert len(event.query_preview or "") <= LIBRARY_ACTIVITY_QUERY_PREVIEW_MAX_CHARS
    assert len(event.source_refs) == LIBRARY_ACTIVITY_SOURCE_REF_MAX_COUNT
    assert all(
        len(ref.source_id) <= LIBRARY_ACTIVITY_SOURCE_ID_MAX_CHARS
        and len(ref.title) <= LIBRARY_ACTIVITY_TITLE_MAX_CHARS
        for ref in event.source_refs
    )
    assert len(encoded) <= LIBRARY_ACTIVITY_PAYLOAD_MAX_BYTES
    text = encoded.decode("utf-8")
    assert "PRIVATE BODY" not in text
    assert "PRIVATE SNIPPET" not in text
    assert "PRIVATE EXCERPT" not in text
    assert "/Users/private" not in text
    assert secret not in text


def test_minimize_activity_scrubs_structured_failure_without_exception_text():
    secret = "Bearer abcdefghijklmnopqrstuvwxyz"
    event = minimize_library_activity(
        _candidate(
            result={
                "error": {
                    "code": "storage_error",
                    "message": (
                        "sqlite failed at /Users/private/library.db " + secret
                    ),
                    "retryable": True,
                    "details": {"traceback": "PRIVATE TRACEBACK"},
                }
            },
            failure_code="storage_error",
        )
    )

    assert event.status == "failed"
    assert event.result_count == 0
    assert event.error_code == "storage_error"
    assert len(event.error_summary or "") <= LIBRARY_ACTIVITY_ERROR_SUMMARY_MAX_CHARS
    encoded = json.dumps(event.to_payload(), ensure_ascii=False)
    assert "/Users/private" not in encoded
    assert secret not in encoded
    assert "PRIVATE TRACEBACK" not in encoded


def test_minimize_activity_marks_zero_results_empty():
    event = minimize_library_activity(_candidate(result={"items": [], "total": 0}))

    assert event.status == "empty"
    assert event.result_count == 0
    assert event.source_refs == ()


def test_decode_activity_uses_strict_schema_types() -> None:
    event = minimize_library_activity(_candidate(result={"items": [], "total": 0}))
    payload = json.loads(encode_library_activity_event(event))
    payload["result_count"] = True

    with pytest.raises(ValueError, match="Invalid Library activity payload"):
        decode_library_activity_event(json.dumps(payload))
