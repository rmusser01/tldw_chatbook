"""Client routing + schema tests for the scheduled-tasks automation control plane."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api.client import TLDWAPIClient
from tldw_chatbook.tldw_api.scheduled_tasks_automation_schemas import (
    ScheduledTaskAutomationDefinition,
    ScheduledTaskAutomationDefinitionList,
    ScheduledTaskAutomationRunNowResponse,
    ScheduledTaskAuditList,
    ScheduledTaskResult,
    ScheduledTaskResultList,
)


@pytest.mark.asyncio
async def test_list_definitions_routes_to_control_plane_with_pagination(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        return_value={
            "items": [
                {
                    "id": "def-1",
                    "family": "recurring_question",
                    "name": "Morning brief",
                    "lifecycle": "configured",
                    "health": "ready",
                }
            ],
            "total": 1,
            "limit": 25,
            "offset": 0,
            "has_more": False,
        }
    )
    monkeypatch.setattr(client, "_request", mocked)

    result = await client.list_scheduled_task_automation_definitions(
        limit=25, offset=0
    )

    assert mocked.await_args.args[:2] == ("GET", "/api/v1/scheduled-tasks/definitions")
    assert mocked.await_args.kwargs["params"] == {"limit": 25, "offset": 0}
    assert isinstance(result, ScheduledTaskAutomationDefinitionList)
    assert result.items[0].id == "def-1"
    assert result.items[0].name == "Morning brief"
    assert result.total == 1


@pytest.mark.asyncio
async def test_run_definition_now_posts_to_run_route_with_idempotency_header(
    monkeypatch,
):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        return_value={
            "definition_id": "def-1",
            "run_slot_utc": "2026-08-29T12:00:00+00:00/2026-08-29T12:00:00+00:00",
            "job_id": 12345,
            "deduped": False,
        }
    )
    monkeypatch.setattr(client, "_request", mocked)

    result = await client.run_scheduled_task_automation_definition_now(
        "def-1", idempotency_key="idem-1"
    )

    assert mocked.await_args.args[:2] == (
        "POST",
        "/api/v1/scheduled-tasks/definitions/def-1/run",
    )
    assert mocked.await_args.kwargs["headers"] == {"Idempotency-Key": "idem-1"}
    assert isinstance(result, ScheduledTaskAutomationRunNowResponse)
    assert result.definition_id == "def-1"
    assert result.job_id == 12345
    assert result.deduped is False


@pytest.mark.asyncio
async def test_run_definition_now_omits_header_without_idempotency_key(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        return_value={
            "definition_id": "def-1",
            "run_slot_utc": "slot",
            "job_id": None,
            "deduped": True,
        }
    )
    monkeypatch.setattr(client, "_request", mocked)

    result = await client.run_scheduled_task_automation_definition_now("def-1")

    assert mocked.await_args.kwargs["headers"] is None
    assert result.deduped is True
    assert result.job_id is None


def test_definition_schema_tolerates_unknown_enum_values_and_missing_policies():
    # The server owns the lifecycle/health vocabularies; a new value must not
    # break the client (ADR-077 phase 1 contract).
    definition = ScheduledTaskAutomationDefinition.model_validate(
        {
            "id": "def-2",
            "family": "agent_task",
            "name": "Novel",
            "lifecycle": "some_future_lifecycle",
            "health": "some_future_health",
        }
    )
    assert definition.lifecycle == "some_future_lifecycle"
    assert definition.health == "some_future_health"
    assert definition.schedule == {}
    assert definition.approval_policy == {}


@pytest.mark.asyncio
async def test_definition_audit_routes_to_audit_endpoint_with_filters(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        return_value={
            "items": [
                {
                    "id": "evt-1",
                    "definition_id": "def-1",
                    "event_type": "run_succeeded",
                    "actor": "automation:consumer",
                    "summary": "Run succeeded.",
                    "after": {"run_id": "run-1", "status": "succeeded"},
                    "created_at": "2026-08-30T00:30:00Z",
                }
            ],
            "total": 1,
            "has_more": False,
        }
    )
    monkeypatch.setattr(client, "_request", mocked)

    result = await client.list_scheduled_task_automation_definition_audit(
        "def-1", limit=20, offset=40, event_type="run_succeeded"
    )

    assert mocked.await_args.args[:2] == (
        "GET",
        "/api/v1/scheduled-tasks/definitions/def-1/audit",
    )
    assert mocked.await_args.kwargs["params"] == {
        "limit": 20,
        "offset": 40,
        "event_type": "run_succeeded",
    }
    assert isinstance(result, ScheduledTaskAuditList)
    assert result.items[0].event_type == "run_succeeded"
    assert result.items[0].after == {"run_id": "run-1", "status": "succeeded"}
    assert result.total == 1


_RESULTS_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "Scheduling"
    / "fixtures"
    / "server_responses"
    / "automation_results_list.json"
)


@pytest.mark.asyncio
async def test_list_results_routes_to_results_endpoint_with_pagination_only(
    monkeypatch,
):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(return_value=json.loads(_RESULTS_FIXTURE.read_text()))
    monkeypatch.setattr(client, "_request", mocked)

    result = await client.list_scheduled_task_results(limit=25, offset=0)

    assert mocked.await_args.args[:2] == ("GET", "/api/v1/scheduled-tasks/results")
    # definition_id/review_state are unset -- must not appear in params.
    assert mocked.await_args.kwargs["params"] == {"limit": 25, "offset": 0}
    assert isinstance(result, ScheduledTaskResultList)
    assert result.total == 2
    assert result.items[0].id == "res_01J5RHPQWXYZ1234567890AB"
    assert result.items[0].review_state == "unread"
    assert result.items[0].answer_mode == "synthesized"
    assert result.items[1].review_state == "read"
    assert result.items[1].reviewed_by == "user:42"


@pytest.mark.asyncio
async def test_list_results_includes_only_the_filters_that_are_set(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(return_value={"items": [], "total": 0})
    monkeypatch.setattr(client, "_request", mocked)

    await client.list_scheduled_task_results(
        definition_id="def-1", review_state="unread"
    )

    assert mocked.await_args.kwargs["params"] == {
        "limit": 50,
        "offset": 0,
        "definition_id": "def-1",
        "review_state": "unread",
    }


@pytest.mark.asyncio
async def test_review_result_posts_to_review_route(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        return_value={
            "id": "res-1",
            "definition_id": "def-1",
            "run_id": "run-1",
            "kind": "finding",
            "title": "T",
            "summary": "S",
            "dedupe_key": "dk-1",
            "review_state": "dismissed",
        }
    )
    monkeypatch.setattr(client, "_request", mocked)

    result = await client.review_scheduled_task_result(
        "res-1", review_state="dismissed", review_note="not relevant"
    )

    assert mocked.await_args.args[:2] == (
        "POST",
        "/api/v1/scheduled-tasks/results/res-1/review",
    )
    assert mocked.await_args.kwargs["json_data"] == {
        "review_state": "dismissed",
        "review_note": "not relevant",
    }
    assert isinstance(result, ScheduledTaskResult)
    assert result.review_state == "dismissed"


@pytest.mark.asyncio
async def test_review_result_sends_null_note_when_not_given(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        return_value={
            "id": "res-1",
            "definition_id": "def-1",
            "run_id": "run-1",
            "kind": "finding",
            "title": "T",
            "summary": "S",
            "dedupe_key": "dk-1",
            "review_state": "read",
        }
    )
    monkeypatch.setattr(client, "_request", mocked)

    await client.review_scheduled_task_result("res-1", review_state="read")

    assert mocked.await_args.kwargs["json_data"] == {
        "review_state": "read",
        "review_note": None,
    }
