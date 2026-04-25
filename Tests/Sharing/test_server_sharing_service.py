from unittest.mock import Mock

import pytest

from tldw_chatbook.Sharing_Interop import ServerSharingService
from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError


class FakeSharingClient:
    def __init__(self):
        self.calls = []

    async def create_share_token(self, request_data):
        self.calls.append(("create_share_token", request_data.model_dump(exclude_none=True, mode="json")))
        return {"id": 9, "raw_token": "secret-token"}

    async def list_share_tokens(self):
        self.calls.append(("list_share_tokens",))
        return {"tokens": [], "total": 0}

    async def revoke_share_token(self, token_id):
        self.calls.append(("revoke_share_token", token_id))
        return {"detail": "Token revoked"}

    async def share_workspace(self, workspace_id, request_data):
        self.calls.append(("share_workspace", workspace_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {"id": 7, "workspace_id": workspace_id}

    async def list_shared_with_me(self):
        self.calls.append(("list_shared_with_me",))
        return {"items": [], "total": 0}

    async def preview_public_share(self, token):
        self.calls.append(("preview_public_share", token))
        return {"resource_type": "workspace", "access_level": "view_chat"}


@pytest.mark.asyncio
async def test_server_sharing_service_routes_links_and_permissions_with_policy_actions():
    client = FakeSharingClient()
    policy = Mock()
    service = ServerSharingService(client=client, policy_enforcer=policy)

    created = await service.create_link(resource_type="workspace", resource_id="ws-1")
    links = await service.list_links()
    revoked = await service.revoke_link(9)
    share = await service.share_workspace(workspace_id="ws-1", share_scope_type="team", share_scope_id=3)
    shared = await service.list_shared_with_me()
    preview = await service.inspect_public_link("secret-token")

    assert created["raw_token"] == "secret-token"
    assert links["total"] == 0
    assert revoked["detail"] == "Token revoked"
    assert share["workspace_id"] == "ws-1"
    assert shared["total"] == 0
    assert preview["resource_type"] == "workspace"
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "sharing.links.create.server",
        "sharing.links.list.server",
        "sharing.links.revoke.server",
        "sharing.permissions.configure.server",
        "sharing.links.list.server",
        "sharing.links.inspect.server",
    ]


@pytest.mark.asyncio
async def test_server_sharing_service_hard_stops_denied_ui_policy_decision():
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
    client = FakeSharingClient()
    service = ServerSharingService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.list_links()

    assert exc.value.reason_code == "server_unreachable"
    assert client.calls == []
