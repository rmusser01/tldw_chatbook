from unittest.mock import Mock

import pytest

from tldw_chatbook.Prompt_Management.server_prompt_service import ServerPromptService
from tldw_chatbook.runtime_policy import PolicyDecision, PolicyDeniedError


class FakePromptClient:
    def __init__(self):
        self.calls = []

    async def list_prompts(self, **kwargs):
        self.calls.append(("list_prompts", kwargs))
        return {"prompts": []}

    async def create_prompt(self, request_data):
        self.calls.append(("create_prompt", request_data.model_dump(exclude_none=True, mode="json")))
        return {"id": 7, "name": request_data.name}

    async def preview_prompt(self, request_data):
        self.calls.append(("preview_prompt", request_data.model_dump(exclude_none=True, mode="json")))
        return {"rendered": "Hello Ada"}

    async def update_prompt(self, prompt_id, request_data):
        self.calls.append(("update_prompt", prompt_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {"id": prompt_id, "name": request_data.name}

    async def delete_prompt(self, prompt_id):
        self.calls.append(("delete_prompt", prompt_id))
        return {"deleted": True}

    async def list_prompt_versions(self, prompt_id):
        self.calls.append(("list_prompt_versions", prompt_id))
        return [{"version": 3, "prompt_uuid": "prompt-uuid"}]

    async def restore_prompt_version(self, prompt_id, version):
        self.calls.append(("restore_prompt_version", prompt_id, version))
        return {"id": prompt_id, "uuid": "prompt-uuid", "version": version, "name": "Restored"}


@pytest.mark.asyncio
async def test_server_prompt_service_enforces_policy_actions():
    client = FakePromptClient()
    policy = Mock()
    service = ServerPromptService(client=client, policy_enforcer=policy)

    await service.list_prompts()
    await service.create_prompt(name="Greeting", user_prompt="Hello {{name}}")
    await service.preview_prompt(name="Greeting", user_prompt="Hello {{name}}")
    await service.update_prompt(7, name="Greeting", user_prompt="Hello {{name}}")
    await service.delete_prompt(7)

    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "prompts.list.server",
        "prompts.create.server",
        "prompts.preview.server",
        "prompts.update.server",
        "prompts.delete.server",
    ]


@pytest.mark.asyncio
async def test_server_prompt_service_routes_prompt_version_controls():
    client = FakePromptClient()
    policy = Mock()
    service = ServerPromptService(client=client, policy_enforcer=policy)

    versions = await service.list_prompt_versions("prompt-uuid")
    restored = await service.restore_prompt_version("prompt-uuid", 3)

    assert versions == [{"version": 3, "prompt_uuid": "prompt-uuid"}]
    assert restored["name"] == "Restored"
    assert client.calls[-2:] == [
        ("list_prompt_versions", "prompt-uuid"),
        ("restore_prompt_version", "prompt-uuid", 3),
    ]
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list][-2:] == [
        "prompts.versions.list.server",
        "prompts.versions.restore.server",
    ]


@pytest.mark.asyncio
async def test_server_prompt_service_hard_stops_denied_ui_policy_decision():
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
    client = FakePromptClient()
    service = ServerPromptService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.list_prompts()

    assert exc.value.reason_code == "server_unreachable"
    assert client.calls == []
