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
    ScheduledTaskDefinitionCreateRequest,
    ScheduledTaskDefinitionUpdateRequest,
    ScheduledTaskPreview,
    ScheduledTaskPreviewCreateRequest,
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


_PREVIEW_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "Scheduling"
    / "fixtures"
    / "server_responses"
    / "automation_preview_response.json"
)


def _load_preview_fixture_case(case_name: str) -> dict:
    return json.loads(_PREVIEW_FIXTURE.read_text())[case_name]


@pytest.mark.asyncio
async def test_preview_definition_posts_to_previews_route(monkeypatch):
    case = _load_preview_fixture_case("valid_recurring_question_create")
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(return_value=case["response"])
    monkeypatch.setattr(client, "_request", mocked)

    # The fixture's `request` half models the local preview port's payload
    # (where `visibility_policy` is `Any`-typed and sent as an explicit
    # `null`), not the server's wire schema -- the wire schema types it as a
    # plain non-nullable `dict[str, Any]` defaulting to `{}` per
    # automation_endpoints.md, so drop that key here and let the model
    # default apply instead of feeding the fixture's `null` into it.
    request_payload = {k: v for k, v in case["request"].items() if k != "visibility_policy"}
    request = ScheduledTaskPreviewCreateRequest(**request_payload)
    result = await client.preview_scheduled_task_definition(request)

    assert mocked.await_args.args[:2] == (
        "POST",
        "/api/v1/scheduled-tasks/previews",
    )
    assert mocked.await_args.kwargs["json_data"]["family"] == "recurring_question"
    assert mocked.await_args.kwargs["json_data"]["name"] == "Daily stand-up summary"
    # visibility_policy defaults to {} per the endpoints doc -- present
    # (exclude_none only drops explicit None, not an empty-dict default).
    assert mocked.await_args.kwargs["json_data"]["visibility_policy"] == {}

    assert isinstance(result, ScheduledTaskPreview)
    assert result.status == "valid"
    assert result.mode == "create"
    assert result.family == "recurring_question"
    assert result.schedule_preview == case["response"]["schedule_preview"]
    assert result.normalized_config == case["response"]["normalized_config"]
    # Round-trip-only fields are absent from the (pure-local-preview) fixture
    # and default to None on the client model.
    assert result.id is None
    assert result.payload_hash is None


@pytest.mark.asyncio
async def test_preview_definition_excludes_unset_optional_fields_from_request_body(
    monkeypatch,
):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        return_value={
            "mode": "create",
            "family": "recurring_question",
            "status": "invalid",
            "validation_errors": [],
        }
    )
    monkeypatch.setattr(client, "_request", mocked)

    request = ScheduledTaskPreviewCreateRequest(family="recurring_question")
    await client.preview_scheduled_task_definition(request)

    body = mocked.await_args.kwargs["json_data"]
    assert body["mode"] == "create"
    assert body["family"] == "recurring_question"
    assert body["config"] == {}
    assert body["schedule"] == {}
    for optional_field in ("definition_id", "definition_version", "name", "description"):
        assert optional_field not in body


def test_preview_schema_validates_fixture_response_for_the_invalid_case():
    case = _load_preview_fixture_case("invalid_recurring_question_bad_schedule_kind")
    preview = ScheduledTaskPreview.model_validate(case["response"])
    assert preview.status == "invalid"
    assert preview.validation_errors == [
        {
            "field": "schedule.kind",
            "code": "unsupported",
            "message": "Unsupported schedule kind: monthly",
        }
    ]


@pytest.mark.asyncio
async def test_create_definition_posts_to_definitions_route(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        return_value={
            "id": "def-1",
            "family": "recurring_question",
            "name": "Daily stand-up summary",
            "lifecycle": "paused",
            "health": "ready",
            "preview_id": "prev-1",
        }
    )
    monkeypatch.setattr(client, "_request", mocked)

    result = await client.create_scheduled_task_definition(
        "prev-1", initial_lifecycle="paused"
    )

    assert mocked.await_args.args[:2] == (
        "POST",
        "/api/v1/scheduled-tasks/definitions",
    )
    assert mocked.await_args.kwargs["json_data"] == {
        "preview_id": "prev-1",
        "initial_lifecycle": "paused",
    }
    assert isinstance(result, ScheduledTaskAutomationDefinition)
    assert result.id == "def-1"
    assert result.lifecycle == "paused"
    assert result.preview_id == "prev-1"


@pytest.mark.asyncio
async def test_create_definition_defaults_lifecycle_to_configured(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        return_value={
            "id": "def-2",
            "family": "recurring_question",
            "name": "N",
            "lifecycle": "configured",
        }
    )
    monkeypatch.setattr(client, "_request", mocked)

    await client.create_scheduled_task_definition("prev-2")

    assert mocked.await_args.kwargs["json_data"] == {
        "preview_id": "prev-2",
        "initial_lifecycle": "configured",
    }


@pytest.mark.asyncio
async def test_update_definition_patches_definitions_route_with_id(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        return_value={
            "id": "def-1",
            "family": "recurring_question",
            "name": "Renamed",
            "lifecycle": "configured",
            "version": 2,
            "preview_id": "prev-3",
        }
    )
    monkeypatch.setattr(client, "_request", mocked)

    result = await client.update_scheduled_task_definition("def-1", "prev-3")

    assert mocked.await_args.args[:2] == (
        "PATCH",
        "/api/v1/scheduled-tasks/definitions/def-1",
    )
    assert mocked.await_args.kwargs["json_data"] == {"preview_id": "prev-3"}
    assert isinstance(result, ScheduledTaskAutomationDefinition)
    assert result.version == 2
    assert result.name == "Renamed"


def test_definition_create_and_update_request_schemas_round_trip():
    create = ScheduledTaskDefinitionCreateRequest(preview_id="prev-1")
    assert create.initial_lifecycle == "configured"
    update = ScheduledTaskDefinitionUpdateRequest(preview_id="prev-2")
    assert update.preview_id == "prev-2"


# ----------------------------------------------------------------------
# Definition lifecycle (pause/resume/archive) -- schedules-handoff PR-5,
# task 2. Mirrors run-now's bare-POST construction (no request body); the
# response schema is the same `ScheduledTaskDefinitionResponse` create/
# update/list already validate as `ScheduledTaskAutomationDefinition`.
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pause_definition_posts_to_pause_route(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        return_value={
            "id": "def-1",
            "family": "recurring_question",
            "name": "Daily stand-up summary",
            "lifecycle": "paused",
        }
    )
    monkeypatch.setattr(client, "_request", mocked)

    result = await client.pause_scheduled_task_definition("def-1")

    assert mocked.await_args.args[:2] == (
        "POST",
        "/api/v1/scheduled-tasks/definitions/def-1/pause",
    )
    assert isinstance(result, ScheduledTaskAutomationDefinition)
    assert result.id == "def-1"
    assert result.lifecycle == "paused"


@pytest.mark.asyncio
async def test_resume_definition_posts_to_resume_route(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        return_value={
            "id": "def-1",
            "family": "recurring_question",
            "name": "Daily stand-up summary",
            "lifecycle": "configured",
        }
    )
    monkeypatch.setattr(client, "_request", mocked)

    result = await client.resume_scheduled_task_definition("def-1")

    assert mocked.await_args.args[:2] == (
        "POST",
        "/api/v1/scheduled-tasks/definitions/def-1/resume",
    )
    assert isinstance(result, ScheduledTaskAutomationDefinition)
    assert result.lifecycle == "configured"


@pytest.mark.asyncio
async def test_archive_definition_posts_to_archive_route(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        return_value={
            "id": "def-1",
            "family": "recurring_question",
            "name": "Daily stand-up summary",
            "lifecycle": "archived",
            "archived_at": "2026-09-01T00:00:00+00:00",
        }
    )
    monkeypatch.setattr(client, "_request", mocked)

    result = await client.archive_scheduled_task_definition("def-1")

    assert mocked.await_args.args[:2] == (
        "POST",
        "/api/v1/scheduled-tasks/definitions/def-1/archive",
    )
    assert isinstance(result, ScheduledTaskAutomationDefinition)
    assert result.lifecycle == "archived"


# ----------------------------------------------------------------------
# Definition resolution (mark-solved/reopen) -- schedules-handoff PR-6,
# task 2. Same bare-POST-with-a-body construction as previews/create; the
# response is again `ScheduledTaskAutomationDefinition`, now carrying the
# four resolution fields the schema was previously missing three of.
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mark_solved_posts_to_mark_solved_route_with_result_id(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        return_value={
            "id": "def-1",
            "family": "recurring_question",
            "name": "Daily stand-up summary",
            "lifecycle": "configured",
            "resolution_state": "solved",
            "resolved_at": "2026-09-02T00:00:00+00:00",
            "resolved_by": "alice",
            "resolved_result_id": "res-9",
        }
    )
    monkeypatch.setattr(client, "_request", mocked)

    result = await client.mark_scheduled_task_definition_solved(
        "def-1", result_id="res-9"
    )

    assert mocked.await_args.args[:2] == (
        "POST",
        "/api/v1/scheduled-tasks/definitions/def-1/mark-solved",
    )
    assert mocked.await_args.kwargs["json_data"] == {"resolved_result_id": "res-9"}
    assert isinstance(result, ScheduledTaskAutomationDefinition)
    assert result.resolution_state == "solved"
    assert result.resolved_at is not None
    assert result.resolved_by == "alice"
    assert result.resolved_result_id == "res-9"


@pytest.mark.asyncio
async def test_mark_solved_sends_null_result_id_when_not_given(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        return_value={
            "id": "def-1",
            "family": "recurring_question",
            "name": "Daily stand-up summary",
            "lifecycle": "configured",
            "resolution_state": "solved",
        }
    )
    monkeypatch.setattr(client, "_request", mocked)

    await client.mark_scheduled_task_definition_solved("def-1")

    assert mocked.await_args.kwargs["json_data"] == {"resolved_result_id": None}


@pytest.mark.asyncio
async def test_reopen_posts_to_reopen_route_with_defaults(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        return_value={
            "id": "def-1",
            "family": "recurring_question",
            "name": "Daily stand-up summary",
            "lifecycle": "paused",
            "resolution_state": "open",
        }
    )
    monkeypatch.setattr(client, "_request", mocked)

    result = await client.reopen_scheduled_task_definition("def-1")

    assert mocked.await_args.args[:2] == (
        "POST",
        "/api/v1/scheduled-tasks/definitions/def-1/reopen",
    )
    assert mocked.await_args.kwargs["json_data"] == {
        "target_lifecycle": "paused",
        "reason": None,
    }
    assert isinstance(result, ScheduledTaskAutomationDefinition)
    assert result.resolution_state == "open"
    assert result.resolved_at is None
    assert result.resolved_by is None
    assert result.resolved_result_id is None


@pytest.mark.asyncio
async def test_reopen_forwards_target_lifecycle_and_reason(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        return_value={
            "id": "def-1",
            "family": "recurring_question",
            "name": "Daily stand-up summary",
            "lifecycle": "configured",
            "resolution_state": "open",
        }
    )
    monkeypatch.setattr(client, "_request", mocked)

    await client.reopen_scheduled_task_definition(
        "def-1", target_lifecycle="configured", reason="False positive"
    )

    assert mocked.await_args.kwargs["json_data"] == {
        "target_lifecycle": "configured",
        "reason": "False positive",
    }
