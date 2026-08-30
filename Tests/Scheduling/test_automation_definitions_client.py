"""Server-client + notifications-service wiring for automation definitions.

task-18940 slice 2: the scheduling server client surfaces the server's
automation control plane (list + run-now) with the same typed-error
discipline as the reminder methods, and the notifications service gates
both behind the ``scheduler.automations`` policy resource.
"""

from unittest.mock import Mock

import pytest

from tldw_chatbook.Notifications import ServerNotificationsService
from tldw_chatbook.Scheduling.services.server_client import (
    SchedulingServerClient,
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


class AutomationNotificationsService:
    """Stub notifications service implementing the two new methods."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def list_scheduled_automations(self, *, limit=50, offset=0):
        self.calls.append(("list", limit, offset))
        return {
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
        }

    async def run_scheduled_automation_now(
        self, definition_id, *, idempotency_key=None
    ):
        self.calls.append(("run", definition_id, idempotency_key))
        return {
            "definition_id": definition_id,
            "run_slot_utc": "slot-1",
            "job_id": 42,
            "deduped": False,
        }


@pytest.mark.asyncio
async def test_list_automation_definitions_passes_pagination_through():
    inner = AutomationNotificationsService()
    client = SchedulingServerClient(inner)

    result = await client.list_automation_definitions(limit=25, offset=50)

    assert result["total"] == 1
    assert result["items"][0]["id"] == "def-1"
    assert inner.calls == [("list", 25, 50)]


@pytest.mark.asyncio
async def test_run_automation_definition_now_returns_run_reference():
    inner = AutomationNotificationsService()
    client = SchedulingServerClient(inner)

    result = await client.run_automation_definition_now("def-1")

    assert result == {
        "definition_id": "def-1",
        "run_slot_utc": "slot-1",
        "job_id": 42,
        "deduped": False,
    }
    assert inner.calls == [("run", "def-1", None)]


@pytest.mark.asyncio
async def test_run_automation_definition_now_is_not_retried():
    # A retried trigger could enqueue a second run; the wrapper must surface
    # the failure after exactly one attempt instead.
    attempts = {"count": 0}

    class FailingService:
        async def run_scheduled_automation_now(
            self, definition_id, *, idempotency_key=None
        ):
            attempts["count"] += 1
            raise ServerClientServerError("boom")

    client = SchedulingServerClient(FailingService())
    with pytest.raises(ServerClientServerError):
        await client.run_automation_definition_now("def-1")
    assert attempts["count"] == 1


@pytest.mark.asyncio
async def test_automation_methods_require_a_connected_server():
    client = SchedulingServerClient(None)
    with pytest.raises(ServerUnavailableError):
        await client.list_automation_definitions()
    with pytest.raises(ServerUnavailableError):
        await client.run_automation_definition_now("def-1")


@pytest.mark.asyncio
async def test_automation_policy_denial_maps_to_policy_error():
    class DenyingService(AutomationNotificationsService):
        async def list_scheduled_automations(self, **kwargs):
            raise PolicyDeniedError(
                action_id="scheduler.automations.list.server",
                reason_code="server_mode_required",
                user_message="scheduler.automations.list.server requires server mode.",
                effective_source="local",
                authority_owner="server",
            )

    client = SchedulingServerClient(DenyingService())
    with pytest.raises(ServerClientPolicyError):
        await client.list_automation_definitions()


@pytest.mark.asyncio
async def test_notifications_service_enforces_scheduler_automations_actions():
    inner = Mock()

    async def _list(**kwargs):
        inner.calls_list = kwargs
        return _FakeResponse({"items": [], "total": 0})

    async def _run(*args, **kwargs):
        inner.calls_run = (args, kwargs)
        return _FakeResponse(
            {
                "definition_id": "def-1",
                "run_slot_utc": "slot-1",
                "job_id": None,
                "deduped": False,
            }
        )

    inner.list_scheduled_task_automation_definitions = _list
    inner.run_scheduled_task_automation_definition_now = _run

    policy = Mock()
    service = ServerNotificationsService(client=inner, policy_enforcer=policy)

    listed = await service.list_scheduled_automations(limit=10, offset=5)
    run = await service.run_scheduled_automation_now("def-1", idempotency_key="k-1")

    assert listed["total"] == 0
    assert run["definition_id"] == "def-1"
    assert [c.kwargs["action_id"] for c in policy.require_allowed.call_args_list] == [
        "scheduler.automations.list.server",
        "scheduler.automations.launch.server",
    ]
    assert inner.calls_list == {"limit": 10, "offset": 5}
    assert inner.calls_run == (("def-1",), {"idempotency_key": "k-1"})


@pytest.mark.asyncio
async def test_notifications_service_hard_stops_denied_automation_launch():
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
        await service.run_scheduled_automation_now("def-1")


class AuditNotificationsService(AutomationNotificationsService):
    """Stub adding the audit-trail method."""

    async def list_scheduled_automation_audit(
        self, definition_id, *, limit=50, offset=0, event_type=None
    ):
        self.calls.append(("audit", definition_id, limit, offset, event_type))
        return {
            "items": [
                {
                    "id": "evt-1",
                    "definition_id": definition_id,
                    "event_type": "run_timed_out",
                    "after": {"run_id": "run-9", "status": "timed_out"},
                }
            ],
            "total": 1,
        }


@pytest.mark.asyncio
async def test_audit_trail_passes_definition_and_pagination_through():
    inner = AuditNotificationsService()
    client = SchedulingServerClient(inner)

    result = await client.list_automation_definition_audit("def-7", limit=15, offset=30)

    assert result["total"] == 1
    assert result["items"][0]["event_type"] == "run_timed_out"
    assert inner.calls == [("audit", "def-7", 15, 30, None)]


@pytest.mark.asyncio
async def test_audit_trail_requires_a_connected_server():
    client = SchedulingServerClient(None)
    with pytest.raises(ServerUnavailableError):
        await client.list_automation_definition_audit("def-7")


@pytest.mark.asyncio
async def test_notifications_service_enforces_audit_read_policy():
    inner = Mock()

    async def _audit(*args, **kwargs):
        inner.audit_args = (args, kwargs)
        return _FakeResponse({"items": [], "total": 0})

    inner.list_scheduled_task_automation_definition_audit = _audit

    policy = Mock()
    service = ServerNotificationsService(client=inner, policy_enforcer=policy)

    trail = await service.list_scheduled_automation_audit(
        "def-7", limit=10, offset=5, event_type="run_failed"
    )

    assert trail["total"] == 0
    policy.require_allowed.assert_called_once_with(
        action_id="scheduler.automations.list.server"
    )
    assert inner.audit_args == (
        ("def-7",),
        {"limit": 10, "offset": 5, "event_type": "run_failed"},
    )
