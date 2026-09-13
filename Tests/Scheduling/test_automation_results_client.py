"""Server-client + notifications-service wiring for scheduled-task results.

Task 2 of the schedules-handoff PR-3 plan: stacks the slice-2 pattern
(tldw_api client -> ``ServerNotificationsService`` policy gate ->
``SchedulingServerClient`` wrapper) for the results-list and
review-pushback seams that ``SyncEngine`` will consume in Task 4.
"""

from unittest.mock import Mock

import pytest

from tldw_chatbook.Notifications import ServerNotificationsService
from tldw_chatbook.Scheduling.services.server_client import (
    SchedulingServerClient,
    ServerClientNotFoundError,
    ServerClientPolicyError,
    ServerClientServerError,
    ServerUnavailableError,
)
from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError


class _FakeResponse:
    """Minimal stand-in for a pydantic response model."""

    def __init__(self, payload):
        self._payload = payload

    def model_dump(self, mode="json"):
        return dict(self._payload)


class AutomationResultsNotificationsService:
    """Stub notifications service implementing the two new methods."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def list_scheduled_automation_results(
        self, *, limit=50, offset=0, definition_id=None, review_state=None
    ):
        self.calls.append(("list", limit, offset, definition_id, review_state))
        return {
            "items": [
                {
                    "id": "res-1",
                    "definition_id": "def-1",
                    "run_id": "run-1",
                    "kind": "finding",
                    "title": "T",
                    "summary": "S",
                    "dedupe_key": "dk-1",
                    "review_state": "unread",
                }
            ],
            "total": 1,
        }

    async def review_scheduled_automation_result(
        self, result_id, review_state, *, review_note=None
    ):
        self.calls.append(("review", result_id, review_state, review_note))
        return {
            "id": result_id,
            "definition_id": "def-1",
            "run_id": "run-1",
            "kind": "finding",
            "title": "T",
            "summary": "S",
            "dedupe_key": "dk-1",
            "review_state": review_state,
        }


@pytest.mark.asyncio
async def test_list_automation_results_passes_filters_through():
    inner = AutomationResultsNotificationsService()
    client = SchedulingServerClient(inner)

    result = await client.list_automation_results(
        limit=25, offset=50, definition_id="def-1", review_state="unread"
    )

    assert result["total"] == 1
    assert result["items"][0]["id"] == "res-1"
    assert inner.calls == [("list", 25, 50, "def-1", "unread")]


@pytest.mark.asyncio
async def test_review_automation_result_returns_updated_row():
    inner = AutomationResultsNotificationsService()
    client = SchedulingServerClient(inner)

    result = await client.review_automation_result(
        "res-1", "dismissed", review_note="not relevant"
    )

    assert result["review_state"] == "dismissed"
    assert inner.calls == [("review", "res-1", "dismissed", "not relevant")]


@pytest.mark.asyncio
async def test_review_automation_result_is_retried_on_server_error():
    # Replaying the same review state is idempotent server-side -- unlike
    # run-now, a retry here cannot double-fire anything.
    attempts = {"count": 0}

    class FlakyThenOk:
        async def review_scheduled_automation_result(
            self, result_id, review_state, *, review_note=None
        ):
            attempts["count"] += 1
            if attempts["count"] < 2:
                raise ServerClientServerError("boom")
            return {"id": result_id, "review_state": review_state}

    from tldw_chatbook.Scheduling.services.server_client import ServerClientConfig

    client = SchedulingServerClient(
        FlakyThenOk(), config=ServerClientConfig(retry_delay=0.0)
    )
    result = await client.review_automation_result("res-1", "read")

    assert attempts["count"] == 2
    assert result["review_state"] == "read"


@pytest.mark.asyncio
async def test_review_automation_result_not_found_maps_to_typed_error():
    class DeletedService:
        async def review_scheduled_automation_result(
            self, result_id, review_state, *, review_note=None
        ):
            raise ServerClientNotFoundError("gone")

    client = SchedulingServerClient(DeletedService())
    with pytest.raises(ServerClientNotFoundError):
        await client.review_automation_result("res-1", "read")


@pytest.mark.asyncio
async def test_automation_results_methods_require_a_connected_server():
    client = SchedulingServerClient(None)
    with pytest.raises(ServerUnavailableError):
        await client.list_automation_results()
    with pytest.raises(ServerUnavailableError):
        await client.review_automation_result("res-1", "read")


@pytest.mark.asyncio
async def test_automation_results_policy_denial_maps_to_policy_error():
    class DenyingService(AutomationResultsNotificationsService):
        async def list_scheduled_automation_results(self, **kwargs):
            raise PolicyDeniedError(
                action_id="scheduler.automations.list.server",
                reason_code="server_mode_required",
                user_message="scheduler.automations.list.server requires server mode.",
                effective_source="local",
                authority_owner="server",
            )

    client = SchedulingServerClient(DenyingService())
    with pytest.raises(ServerClientPolicyError):
        await client.list_automation_results()


@pytest.mark.asyncio
async def test_notifications_service_gates_list_results_under_the_list_action():
    inner = Mock()

    async def _list(**kwargs):
        inner.calls_list = kwargs
        return _FakeResponse({"items": [], "total": 0})

    inner.list_scheduled_task_results = _list

    policy = Mock()
    service = ServerNotificationsService(client=inner, policy_enforcer=policy)

    listed = await service.list_scheduled_automation_results(
        limit=10, offset=5, definition_id="def-1", review_state="unread"
    )

    assert listed["total"] == 0
    policy.require_allowed.assert_called_once_with(
        action_id="scheduler.automations.list.server"
    )
    assert inner.calls_list == {
        "limit": 10,
        "offset": 5,
        "definition_id": "def-1",
        "review_state": "unread",
    }


@pytest.mark.asyncio
async def test_notifications_service_gates_review_under_the_configure_action():
    inner = Mock()

    async def _review(*args, **kwargs):
        inner.review_args = (args, kwargs)
        return _FakeResponse(
            {
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

    inner.review_scheduled_task_result = _review

    policy = Mock()
    service = ServerNotificationsService(client=inner, policy_enforcer=policy)

    reviewed = await service.review_scheduled_automation_result(
        "res-1", "dismissed", review_note="not relevant"
    )

    assert reviewed["review_state"] == "dismissed"
    # Reviewing mutates a result (not a list, not a run trigger) -- the
    # configure-class action, same reasoning family as
    # notifications.reminders.configure.server for reminder CRUD.
    policy.require_allowed.assert_called_once_with(
        action_id="scheduler.automations.configure.server"
    )
    assert inner.review_args == (
        ("res-1", "dismissed"),
        {"review_note": "not relevant"},
    )


@pytest.mark.asyncio
async def test_notifications_service_hard_stops_denied_result_review():
    policy = Mock()
    policy.require_allowed = None
    policy.require_ui_action_allowed = Mock(
        return_value=PolicyDecision(
            allowed=False,
            reason_code="authority_denied",
            user_message="Blocked.",
            effective_source="server",
            authority_owner="server",
        )
    )
    service = ServerNotificationsService(client=Mock(), policy_enforcer=policy)
    with pytest.raises(PolicyDeniedError):
        await service.review_scheduled_automation_result("res-1", "read")
