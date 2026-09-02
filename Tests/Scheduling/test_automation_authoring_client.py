"""Server-client + notifications-service wiring for definition authoring.

Task 2 of the schedules-handoff PR-4 plan: stacks the slice-2/PR-3 pattern
(tldw_api client -> ``ServerNotificationsService`` policy gate ->
``SchedulingServerClient`` wrapper) for the preview/create/update seams
(spec §5.1) that ``SyncEngine`` will consume in Task 3 to replay locally
authored definitions to the server.
"""

from unittest.mock import Mock

import pytest

from tldw_chatbook.Notifications import ServerNotificationsService
from tldw_chatbook.Scheduling.services.server_client import (
    SchedulingServerClient,
    ServerClientConfig,
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


_PREVIEW_REQUEST = {
    "mode": "create",
    "family": "recurring_question",
    "name": "Daily stand-up summary",
    "schedule": {"kind": "daily", "time_of_day": "09:00", "timezone": "UTC"},
}


class AutomationAuthoringNotificationsService:
    """Stub notifications service implementing the three new methods."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def preview_scheduled_automation_definition(self, request):
        self.calls.append(("preview", request))
        return {
            "mode": request.get("mode", "create"),
            "family": request["family"],
            "status": "valid",
            "validation_errors": [],
            "normalized_config": {"family": request["family"]},
        }

    async def create_scheduled_automation_definition(
        self, preview_id, *, initial_lifecycle="configured"
    ):
        self.calls.append(("create", preview_id, initial_lifecycle))
        return {
            "id": "def-1",
            "family": "recurring_question",
            "name": "Daily stand-up summary",
            "lifecycle": initial_lifecycle,
            "preview_id": preview_id,
        }

    async def update_scheduled_automation_definition(self, definition_id, preview_id):
        self.calls.append(("update", definition_id, preview_id))
        return {
            "id": definition_id,
            "family": "recurring_question",
            "name": "Renamed",
            "lifecycle": "configured",
            "preview_id": preview_id,
        }

    async def pause_scheduled_automation_definition(self, definition_id):
        self.calls.append(("pause", definition_id))
        return {"id": definition_id, "family": "recurring_question", "lifecycle": "paused"}

    async def resume_scheduled_automation_definition(self, definition_id):
        self.calls.append(("resume", definition_id))
        return {"id": definition_id, "family": "recurring_question", "lifecycle": "configured"}

    async def archive_scheduled_automation_definition(self, definition_id):
        self.calls.append(("archive", definition_id))
        return {"id": definition_id, "family": "recurring_question", "lifecycle": "archived"}


@pytest.mark.asyncio
async def test_preview_automation_definition_passes_payload_through():
    inner = AutomationAuthoringNotificationsService()
    client = SchedulingServerClient(inner)

    result = await client.preview_automation_definition(_PREVIEW_REQUEST)

    assert result["status"] == "valid"
    assert result["family"] == "recurring_question"
    assert inner.calls == [("preview", _PREVIEW_REQUEST)]


@pytest.mark.asyncio
async def test_create_automation_definition_passes_preview_id_and_lifecycle():
    inner = AutomationAuthoringNotificationsService()
    client = SchedulingServerClient(inner)

    result = await client.create_automation_definition(
        "prev-1", initial_lifecycle="paused"
    )

    assert result["id"] == "def-1"
    assert result["lifecycle"] == "paused"
    assert inner.calls == [("create", "prev-1", "paused")]


@pytest.mark.asyncio
async def test_create_automation_definition_defaults_lifecycle_to_configured():
    inner = AutomationAuthoringNotificationsService()
    client = SchedulingServerClient(inner)

    await client.create_automation_definition("prev-2")

    assert inner.calls == [("create", "prev-2", "configured")]


@pytest.mark.asyncio
async def test_update_automation_definition_passes_definition_and_preview_ids():
    inner = AutomationAuthoringNotificationsService()
    client = SchedulingServerClient(inner)

    result = await client.update_automation_definition("def-1", "prev-3")

    assert result["id"] == "def-1"
    assert result["preview_id"] == "prev-3"
    assert inner.calls == [("update", "def-1", "prev-3")]


@pytest.mark.asyncio
async def test_preview_and_create_are_retried_on_server_error():
    # Replaying a preview or a create-from-preview hits the server's
    # payload-hash idempotency, so a transient failure is safe to retry
    # (spec §5.1) -- unlike run-now, it cannot double-create anything.
    attempts = {"preview": 0, "create": 0}

    class FlakyThenOk:
        async def preview_scheduled_automation_definition(self, request):
            attempts["preview"] += 1
            if attempts["preview"] < 2:
                raise ServerClientServerError("boom")
            return {"status": "valid"}

        async def create_scheduled_automation_definition(
            self, preview_id, *, initial_lifecycle="configured"
        ):
            attempts["create"] += 1
            if attempts["create"] < 2:
                raise ServerClientServerError("boom")
            return {"id": "def-1"}

    client = SchedulingServerClient(
        FlakyThenOk(), config=ServerClientConfig(retry_delay=0.0)
    )

    preview_result = await client.preview_automation_definition(_PREVIEW_REQUEST)
    create_result = await client.create_automation_definition("prev-1")

    assert attempts["preview"] == 2
    assert attempts["create"] == 2
    assert preview_result["status"] == "valid"
    assert create_result["id"] == "def-1"


@pytest.mark.asyncio
async def test_update_automation_definition_is_not_retried():
    """Qodo HIGH: the update PATCH consumes its preview, so a transport-level
    retry after an unobserved success gets the preview-already-consumed 409
    back and the save reports "the server refused your edit" for an edit the
    server actually APPLIED. One attempt only -- an ambiguous transport
    failure then falls through to `save_definition`'s offline queue, whose
    replay takes a FRESH update-mode preview (the only safe retry here)."""
    attempts = {"count": 0}

    class AlwaysFailing:
        async def update_scheduled_automation_definition(self, definition_id, preview_id):
            attempts["count"] += 1
            raise ServerClientServerError("boom")

    client = SchedulingServerClient(
        AlwaysFailing(), config=ServerClientConfig(retry_delay=0.0)
    )

    with pytest.raises(ServerClientServerError):
        await client.update_automation_definition("def-1", "prev-1")

    assert attempts["count"] == 1


@pytest.mark.asyncio
async def test_update_automation_definition_not_found_maps_to_typed_error():
    class DeletedService:
        async def update_scheduled_automation_definition(self, definition_id, preview_id):
            raise ServerClientNotFoundError("gone")

    client = SchedulingServerClient(DeletedService())
    with pytest.raises(ServerClientNotFoundError):
        await client.update_automation_definition("def-1", "prev-1")


@pytest.mark.asyncio
async def test_automation_authoring_methods_require_a_connected_server():
    client = SchedulingServerClient(None)
    with pytest.raises(ServerUnavailableError):
        await client.preview_automation_definition(_PREVIEW_REQUEST)
    with pytest.raises(ServerUnavailableError):
        await client.create_automation_definition("prev-1")
    with pytest.raises(ServerUnavailableError):
        await client.update_automation_definition("def-1", "prev-1")


@pytest.mark.asyncio
async def test_automation_authoring_policy_denial_maps_to_policy_error():
    class DenyingService(AutomationAuthoringNotificationsService):
        async def preview_scheduled_automation_definition(self, request):
            raise PolicyDeniedError(
                action_id="scheduler.automations.configure.server",
                reason_code="server_mode_required",
                user_message="scheduler.automations.configure.server requires server mode.",
                effective_source="local",
                authority_owner="server",
            )

    client = SchedulingServerClient(DenyingService())
    with pytest.raises(ServerClientPolicyError):
        await client.preview_automation_definition(_PREVIEW_REQUEST)


@pytest.mark.asyncio
async def test_pause_automation_definition_passes_definition_id():
    inner = AutomationAuthoringNotificationsService()
    client = SchedulingServerClient(inner)

    result = await client.pause_automation_definition("def-1")

    assert result["lifecycle"] == "paused"
    assert inner.calls == [("pause", "def-1")]


@pytest.mark.asyncio
async def test_resume_automation_definition_passes_definition_id():
    inner = AutomationAuthoringNotificationsService()
    client = SchedulingServerClient(inner)

    result = await client.resume_automation_definition("def-1")

    assert result["lifecycle"] == "configured"
    assert inner.calls == [("resume", "def-1")]


@pytest.mark.asyncio
async def test_archive_automation_definition_passes_definition_id():
    inner = AutomationAuthoringNotificationsService()
    client = SchedulingServerClient(inner)

    result = await client.archive_automation_definition("def-1")

    assert result["lifecycle"] == "archived"
    assert inner.calls == [("archive", "def-1")]


@pytest.mark.asyncio
async def test_lifecycle_seams_are_retried_on_server_error():
    # Lifecycle transitions are idempotent by nature -- pausing an
    # already-paused definition is a no-op server-side -- so, unlike
    # run-now/update, these keep the default retry behavior.
    attempts = {"pause": 0}

    class FlakyThenOk:
        async def pause_scheduled_automation_definition(self, definition_id):
            attempts["pause"] += 1
            if attempts["pause"] < 2:
                raise ServerClientServerError("boom")
            return {"id": definition_id, "lifecycle": "paused"}

    client = SchedulingServerClient(
        FlakyThenOk(), config=ServerClientConfig(retry_delay=0.0)
    )

    result = await client.pause_automation_definition("def-1")

    assert attempts["pause"] == 2
    assert result["lifecycle"] == "paused"


@pytest.mark.asyncio
async def test_lifecycle_seam_not_found_maps_to_typed_error():
    class DeletedService:
        async def archive_scheduled_automation_definition(self, definition_id):
            raise ServerClientNotFoundError("gone")

    client = SchedulingServerClient(DeletedService())
    with pytest.raises(ServerClientNotFoundError):
        await client.archive_automation_definition("def-1")


@pytest.mark.asyncio
async def test_lifecycle_seams_require_a_connected_server():
    client = SchedulingServerClient(None)
    with pytest.raises(ServerUnavailableError):
        await client.pause_automation_definition("def-1")
    with pytest.raises(ServerUnavailableError):
        await client.resume_automation_definition("def-1")
    with pytest.raises(ServerUnavailableError):
        await client.archive_automation_definition("def-1")


@pytest.mark.asyncio
async def test_lifecycle_seam_policy_denial_maps_to_policy_error():
    class DenyingService(AutomationAuthoringNotificationsService):
        async def pause_scheduled_automation_definition(self, definition_id):
            raise PolicyDeniedError(
                action_id="scheduler.automations.configure.server",
                reason_code="server_mode_required",
                user_message="scheduler.automations.configure.server requires server mode.",
                effective_source="local",
                authority_owner="server",
            )

    client = SchedulingServerClient(DenyingService())
    with pytest.raises(ServerClientPolicyError):
        await client.pause_automation_definition("def-1")


@pytest.mark.asyncio
async def test_notifications_service_gates_lifecycle_seams_under_configure_action():
    inner = Mock()

    async def _pause(definition_id):
        inner.pause_args = definition_id
        return _FakeResponse({"id": definition_id, "lifecycle": "paused"})

    async def _resume(definition_id):
        inner.resume_args = definition_id
        return _FakeResponse({"id": definition_id, "lifecycle": "configured"})

    async def _archive(definition_id):
        inner.archive_args = definition_id
        return _FakeResponse({"id": definition_id, "lifecycle": "archived"})

    inner.pause_scheduled_task_definition = _pause
    inner.resume_scheduled_task_definition = _resume
    inner.archive_scheduled_task_definition = _archive

    policy = Mock()
    service = ServerNotificationsService(client=inner, policy_enforcer=policy)

    paused = await service.pause_scheduled_automation_definition("def-1")
    resumed = await service.resume_scheduled_automation_definition("def-1")
    archived = await service.archive_scheduled_automation_definition("def-1")

    assert paused["lifecycle"] == "paused"
    assert resumed["lifecycle"] == "configured"
    assert archived["lifecycle"] == "archived"
    assert [c.kwargs["action_id"] for c in policy.require_allowed.call_args_list] == [
        "scheduler.automations.configure.server",
        "scheduler.automations.configure.server",
        "scheduler.automations.configure.server",
    ]
    assert inner.pause_args == "def-1"
    assert inner.resume_args == "def-1"
    assert inner.archive_args == "def-1"


@pytest.mark.asyncio
async def test_notifications_service_gates_all_three_seams_under_configure_action():
    inner = Mock()

    async def _preview(request):
        inner.preview_args = request
        return _FakeResponse({"status": "valid", "family": request.family})

    async def _create(preview_id, *, initial_lifecycle="configured"):
        inner.create_args = (preview_id, initial_lifecycle)
        return _FakeResponse({"id": "def-1", "lifecycle": initial_lifecycle})

    async def _update(definition_id, preview_id):
        inner.update_args = (definition_id, preview_id)
        return _FakeResponse({"id": definition_id, "preview_id": preview_id})

    inner.preview_scheduled_task_definition = _preview
    inner.create_scheduled_task_definition = _create
    inner.update_scheduled_task_definition = _update

    policy = Mock()
    service = ServerNotificationsService(client=inner, policy_enforcer=policy)

    previewed = await service.preview_scheduled_automation_definition(_PREVIEW_REQUEST)
    created = await service.create_scheduled_automation_definition(
        "prev-1", initial_lifecycle="paused"
    )
    updated = await service.update_scheduled_automation_definition("def-1", "prev-2")

    assert previewed["status"] == "valid"
    assert created["lifecycle"] == "paused"
    assert updated["preview_id"] == "prev-2"
    assert [c.kwargs["action_id"] for c in policy.require_allowed.call_args_list] == [
        "scheduler.automations.configure.server",
        "scheduler.automations.configure.server",
        "scheduler.automations.configure.server",
    ]
    # The service builds a real ScheduledTaskPreviewCreateRequest from the
    # raw dict before calling the typed client method.
    assert inner.preview_args.family == "recurring_question"
    assert inner.preview_args.name == "Daily stand-up summary"
    assert inner.create_args == ("prev-1", "paused")
    assert inner.update_args == ("def-1", "prev-2")


@pytest.mark.asyncio
async def test_notifications_service_hard_stops_denied_preview():
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
        await service.preview_scheduled_automation_definition(_PREVIEW_REQUEST)
