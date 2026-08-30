"""Client routing + schema tests for the scheduled-tasks automation control plane."""

from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api.client import TLDWAPIClient
from tldw_chatbook.tldw_api.scheduled_tasks_automation_schemas import (
    ScheduledTaskAutomationDefinition,
    ScheduledTaskAutomationDefinitionList,
    ScheduledTaskAutomationRunNowResponse,
    ScheduledTaskAuditList,
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
