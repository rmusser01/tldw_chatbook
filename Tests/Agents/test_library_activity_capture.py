"""Trusted Direct/RAG provider activity-capture boundary tests."""

from __future__ import annotations

import json

from tldw_chatbook.Agents.library_rag_tool_provider import (
    LibraryRagToolProvider,
    RAG_TOOL_NAME,
)
from tldw_chatbook.Agents.library_tool_provider import LibraryToolProvider
from tldw_chatbook.Agents.run_context import CurrentRunActor, use_run_actor
from tldw_chatbook.Library.library_rag_state import LibraryRagResultRow


class _DirectService:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def invoke(self, _name: str, _arguments: object) -> object:
        return self.payload


class _RagService:
    def __init__(self, rows: list[LibraryRagResultRow]) -> None:
        self.rows = rows

    async def search(self, *_args, **_kwargs):
        return {"results": self.rows}


def _row(index: int) -> LibraryRagResultRow:
    return LibraryRagResultRow.from_result(
        {
            "result_id": f"row-{index}",
            "source_id": f"note-{index}",
            "title": f"Title {index} " + "界" * 2_000,
            "snippet": "body " + "x" * 6_000,
            "score": 1.0,
            "source_type": "note",
        }
    )


def test_direct_provider_captures_primary_activity_before_returning_result():
    events = []
    provider = LibraryToolProvider(
        _DirectService({"items": [{"id": "note:1", "type": "note", "title": "One"}]}),
        activity_attempt_id="attempt-1",
        activity_sink=events.append,
    )

    with use_run_actor(CurrentRunActor("primary", "run-1", None)):
        result = provider.invoke("library:library_search_notes", {"query": "one"})

    assert result.ok is True
    assert len(events) == 1
    event = events[0]
    assert event.attempt_id == "attempt-1"
    assert event.actor_kind == "primary"
    assert event.run_id == "run-1"
    assert event.parent_run_id is None
    assert event.library_provider == "direct"
    assert event.operation == "library_search_notes"


def test_rag_capture_sees_authoritative_rows_before_model_payload_bounding():
    events = []
    rows = [_row(index) for index in range(10)]
    provider = LibraryRagToolProvider(
        _RagService(rows),
        activity_attempt_id="attempt-rag",
        activity_sink=events.append,
    )

    with use_run_actor(CurrentRunActor("subagent", "run-child", "run-parent")):
        result = provider.invoke(
            f"library:{RAG_TOOL_NAME}", {"query": "needle", "top_k": 10}
        )

    assert result.ok is True
    assert len(events) == 1
    assert len(events[0].source_refs) == 8
    assert len(json.loads(result.content)["results"]) < len(rows)
    assert events[0].actor_kind == "subagent"
    assert events[0].parent_run_id == "run-parent"


def test_capture_failure_withholds_direct_library_payload():
    def reject(_event) -> None:
        raise RuntimeError("must not escape")

    provider = LibraryToolProvider(
        _DirectService({"items": [{"id": "note:secret", "body": "SECRET BODY"}]}),
        activity_attempt_id="attempt-1",
        activity_sink=reject,
    )

    with use_run_actor(CurrentRunActor("primary", "run-1", None)):
        result = provider.invoke("library:library_search_notes", {"query": "one"})

    assert result.ok is False
    assert "SECRET BODY" not in result.error
    error = json.loads(result.error)["error"]
    assert error["code"] == "storage_error"
    assert error["message"] == (
        "Library result withheld because activity could not be recorded."
    )
    assert error["details"] == {"category": "review_capture_failed"}


def test_configured_capture_without_bound_actor_fails_closed():
    provider = LibraryToolProvider(
        _DirectService({"items": [{"id": "note:1"}]}),
        activity_attempt_id="attempt-1",
        activity_sink=lambda _event: None,
    )

    result = provider.invoke("library:library_search_notes", {"query": "one"})

    assert result.ok is False
    assert "could not be recorded" in result.error


def test_attempts_remain_distinct_for_the_same_subagent_run():
    events = []
    actor = CurrentRunActor("subagent", "run-child", "run-parent")
    with use_run_actor(actor):
        for attempt_id in ("attempt-1", "attempt-2"):
            provider = LibraryToolProvider(
                _DirectService({"items": []}),
                activity_attempt_id=attempt_id,
                activity_sink=events.append,
            )
            assert provider.invoke("library:library_search_notes", {"query": "q"}).ok

    assert [event.attempt_id for event in events] == ["attempt-1", "attempt-2"]
    assert events[0].event_id != events[1].event_id


def test_rag_provider_captures_refused_valid_operation():
    events = []
    provider = LibraryRagToolProvider(
        _RagService([]),
        activity_attempt_id="attempt-rag",
        activity_sink=events.append,
    )

    with use_run_actor(CurrentRunActor("primary", "run-1", None)):
        result = provider.invoke(
            f"library:{RAG_TOOL_NAME}", {"query": "needle", "top_k": 0}
        )

    assert result.ok is False
    assert len(events) == 1
    assert events[0].status == "failed"
    assert events[0].error_code == "invalid_argument"
    assert events[0].query_preview == "needle"
