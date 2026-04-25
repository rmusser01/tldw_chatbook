from unittest.mock import Mock

import pytest

from tldw_chatbook.Outputs_Interop import ServerOutputsService
from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError


class FakeOutputsClient:
    def __init__(self):
        self.calls = []

    async def list_output_templates(self, **kwargs):
        self.calls.append(("list_output_templates", kwargs))
        return type("Response", (), {"model_dump": lambda self, mode="json": {"items": [], "total": 0}})()

    async def create_output_template(self, request_data):
        self.calls.append(("create_output_template", request_data.model_dump(exclude_none=True, mode="json")))
        return {"id": 1, "name": "Newsletter"}

    async def preview_output_template(self, template_id, request_data):
        self.calls.append(("preview_output_template", template_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {"rendered": "# Digest", "format": "md"}

    async def list_outputs(self, **kwargs):
        self.calls.append(("list_outputs", kwargs))
        return {"items": [], "total": 0, "page": 1, "size": 50}

    async def create_output(self, request_data):
        self.calls.append(("create_output", request_data.model_dump(exclude_none=True, mode="json")))
        return {"id": 9, "title": "Digest"}

    async def delete_output(self, output_id, **kwargs):
        self.calls.append(("delete_output", output_id, kwargs))
        return {"success": True, "file_deleted": False}


@pytest.mark.asyncio
async def test_server_outputs_service_routes_templates_artifacts_and_render_with_policy_actions():
    client = FakeOutputsClient()
    policy = Mock()
    service = ServerOutputsService(client=client, policy_enforcer=policy)

    templates = await service.list_templates(q="news")
    created_template = await service.create_template(
        name="Newsletter",
        type="newsletter_markdown",
        format="md",
        body="# {{ title }}",
    )
    preview = await service.preview_template(1, data={"title": "Digest"})
    artifacts = await service.list_artifacts(type="newsletter_markdown")
    created_artifact = await service.create_artifact(template_id=1, data={"title": "Digest"})
    deleted_artifact = await service.delete_artifact(9, hard=True)

    assert templates["total"] == 0
    assert created_template["id"] == 1
    assert preview["rendered"] == "# Digest"
    assert artifacts["total"] == 0
    assert created_artifact["id"] == 9
    assert deleted_artifact["success"] is True
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "outputs.templates.list.server",
        "outputs.templates.create.server",
        "outputs.render_jobs.launch.server",
        "outputs.artifacts.list.server",
        "outputs.artifacts.create.server",
        "outputs.artifacts.delete.server",
    ]


@pytest.mark.asyncio
async def test_server_outputs_service_hard_stops_denied_ui_policy_decision():
    policy = Mock()
    policy.require_allowed = None
    policy.require_ui_action_allowed = Mock(
        return_value=PolicyDecision(
            allowed=False,
            reason_code="server_unreachable",
            user_message="Blocked.",
            effective_source="server",
            authority_owner="server",
        )
    )
    client = FakeOutputsClient()
    service = ServerOutputsService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.list_templates()

    assert exc.value.reason_code == "server_unreachable"
    assert client.calls == []
