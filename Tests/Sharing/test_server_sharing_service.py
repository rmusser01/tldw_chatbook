from __future__ import annotations

from typing import Any

import pytest

from tldw_chatbook.Sharing import ServerSharingScopeService, ServerSharingService


class FakePolicyEnforcer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def require_allowed(self, *, action_id: str) -> None:
        self.calls.append(action_id)


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    async def share_workspace(self, workspace_id, request_data):
        payload = request_data.model_dump(mode="json")
        self.calls.append(("share_workspace", workspace_id, payload))
        return {
            "id": 7,
            "workspace_id": workspace_id,
            "owner_user_id": 3,
            "share_scope_type": payload["share_scope_type"],
            "share_scope_id": payload["share_scope_id"],
            "access_level": payload["access_level"],
            "allow_clone": payload["allow_clone"],
            "created_by": 3,
            "created_at": "2026-04-23T20:00:00Z",
            "updated_at": None,
            "revoked_at": None,
            "is_revoked": False,
        }

    async def list_workspace_shares(self, workspace_id, *, include_revoked=False):
        self.calls.append(("list_workspace_shares", workspace_id, include_revoked))
        return {
            "shares": [
                {
                    "id": 7,
                    "workspace_id": workspace_id,
                    "owner_user_id": 3,
                    "share_scope_type": "team",
                    "share_scope_id": 11,
                    "access_level": "view_chat",
                    "allow_clone": True,
                    "created_by": 3,
                    "created_at": "2026-04-23T20:00:00Z",
                    "updated_at": None,
                    "revoked_at": None,
                    "is_revoked": False,
                }
            ],
            "total": 1,
        }

    async def update_share(self, share_id, request_data):
        payload = request_data.model_dump(exclude_none=True, mode="json")
        self.calls.append(("update_share", share_id, payload))
        return {
            "id": share_id,
            "workspace_id": "ws-1",
            "owner_user_id": 3,
            "share_scope_type": "team",
            "share_scope_id": 11,
            "access_level": payload["access_level"],
            "allow_clone": payload["allow_clone"],
            "created_by": 3,
            "created_at": "2026-04-23T20:00:00Z",
            "updated_at": "2026-04-23T21:00:00Z",
            "revoked_at": None,
            "is_revoked": False,
        }

    async def revoke_share(self, share_id):
        self.calls.append(("revoke_share", share_id))
        return {"detail": "Share revoked"}

    async def list_shared_with_me(self):
        self.calls.append(("list_shared_with_me",))
        return {
            "items": [
                {
                    "share_id": 7,
                    "workspace_id": "ws-1",
                    "workspace_name": "Shared Workspace",
                    "owner_user_id": 3,
                    "owner_username": "owner",
                    "access_level": "view_chat",
                    "allow_clone": True,
                    "shared_at": "2026-04-23T20:00:00Z",
                }
            ],
            "total": 1,
        }

    async def create_share_token(self, request_data):
        payload = request_data.model_dump(exclude_none=True, mode="json")
        self.calls.append(("create_share_token", payload))
        return {
            "id": 5,
            "token_prefix": "abc123",
            "resource_type": payload["resource_type"],
            "resource_id": payload["resource_id"],
            "access_level": payload["access_level"],
            "allow_clone": payload["allow_clone"],
            "is_password_protected": bool(payload.get("password")),
            "max_uses": payload.get("max_uses"),
            "use_count": 0,
            "expires_at": payload.get("expires_at"),
            "created_at": "2026-04-23T20:00:00Z",
            "revoked_at": None,
            "is_revoked": False,
            "raw_token": "raw-token",
        }

    async def clone_shared_workspace(self, share_id, request_data):
        payload = request_data.model_dump(exclude_none=True, mode="json")
        self.calls.append(("clone_shared_workspace", share_id, payload))
        return {"job_id": "clone-job-1", "status": "pending", "message": "Clone job created"}

    async def chat_with_shared_workspace(self, share_id, request_data):
        payload = request_data.model_dump(exclude_none=True, mode="json")
        self.calls.append(("chat_with_shared_workspace", share_id, payload))
        return {"result": {"answer": "Shared answer"}}

    async def preview_public_share(self, token):
        self.calls.append(("preview_public_share", token))
        return {
            "resource_type": "workspace",
            "resource_name": "Shared Workspace",
            "resource_description": None,
            "is_password_protected": False,
            "access_level": "view_chat",
        }


@pytest.mark.asyncio
async def test_server_sharing_service_routes_typed_client_calls():
    client = FakeClient()
    service = ServerSharingService(client=client)

    share = await service.share_workspace(
        "ws-1",
        share_scope_type="team",
        share_scope_id=11,
        access_level="view_chat",
        allow_clone=True,
    )
    shares = await service.list_workspace_shares("ws-1", include_revoked=True)
    updated = await service.update_share(7, access_level="full_edit", allow_clone=False)
    revoked = await service.revoke_share(7)
    shared_with_me = await service.list_shared_with_me()
    token = await service.create_share_token(
        resource_type="workspace",
        resource_id="ws-1",
        access_level="view_chat",
        allow_clone=True,
    )
    clone = await service.clone_shared_workspace(7, new_name="My Copy")
    chat = await service.chat_with_shared_workspace(7, query="Summarize")
    preview = await service.preview_public_share("raw-token")

    assert share["id"] == 7
    assert shares["total"] == 1
    assert updated["access_level"] == "full_edit"
    assert revoked == {"detail": "Share revoked"}
    assert shared_with_me["items"][0]["share_id"] == 7
    assert token["raw_token"] == "raw-token"
    assert clone["job_id"] == "clone-job-1"
    assert chat["result"]["answer"] == "Shared answer"
    assert preview["resource_type"] == "workspace"
    assert client.calls == [
        ("share_workspace", "ws-1", {"share_scope_type": "team", "share_scope_id": 11, "access_level": "view_chat", "allow_clone": True}),
        ("list_workspace_shares", "ws-1", True),
        ("update_share", 7, {"access_level": "full_edit", "allow_clone": False}),
        ("revoke_share", 7),
        ("list_shared_with_me",),
        ("create_share_token", {"resource_type": "workspace", "resource_id": "ws-1", "access_level": "view_chat", "allow_clone": True}),
        ("clone_shared_workspace", 7, {"new_name": "My Copy"}),
        ("chat_with_shared_workspace", 7, {"query": "Summarize"}),
        ("preview_public_share", "raw-token"),
    ]


@pytest.mark.asyncio
async def test_sharing_scope_service_enforces_policy_and_normalizes_ids():
    policy = FakePolicyEnforcer()
    scope = ServerSharingScopeService(
        server_service=ServerSharingService(client=FakeClient()),
        policy_enforcer=policy,
    )

    created = await scope.share_workspace(
        mode="server",
        workspace_id="ws-1",
        share_scope_type="team",
        share_scope_id=11,
        access_level="view_chat",
        allow_clone=True,
    )
    listed = await scope.list_workspace_shares(mode="server", workspace_id="ws-1", include_revoked=True)
    updated = await scope.update_share(mode="server", share_id=7, access_level="full_edit", allow_clone=False)
    revoked = await scope.revoke_share(mode="server", share_id=7)
    shared_with_me = await scope.list_shared_with_me(mode="server")
    token = await scope.create_share_token(
        mode="server",
        resource_type="workspace",
        resource_id="ws-1",
        access_level="view_chat",
        allow_clone=True,
    )
    clone = await scope.clone_shared_workspace(mode="server", share_id=7, new_name="My Copy")
    chat = await scope.chat_with_shared_workspace(mode="server", share_id=7, query="Summarize")
    preview = await scope.preview_public_share(mode="server", token="raw-token")

    assert created["id"] == "server:share:7"
    assert created["source_id"] == 7
    assert listed["shares"][0]["id"] == "server:share:7"
    assert updated["id"] == "server:share:7"
    assert revoked["backend"] == "server"
    assert shared_with_me["items"][0]["id"] == "server:share:7"
    assert token["id"] == "server:share_token:5"
    assert clone["entity_kind"] == "share_clone_job"
    assert chat["backend"] == "server"
    assert preview["entity_kind"] == "public_share_preview"
    assert policy.calls == [
        "sharing.permissions.configure.server",
        "sharing.links.list.server",
        "sharing.permissions.configure.server",
        "sharing.links.revoke.server",
        "sharing.links.list.server",
        "sharing.links.create.server",
        "sharing.links.launch.server",
        "sharing.links.launch.server",
        "sharing.links.inspect.server",
    ]


@pytest.mark.asyncio
async def test_sharing_scope_service_rejects_local_mode_before_policy_dispatch():
    policy = FakePolicyEnforcer()
    scope = ServerSharingScopeService(
        server_service=ServerSharingService(client=FakeClient()),
        policy_enforcer=policy,
    )

    with pytest.raises(ValueError, match="Server sharing requires server mode"):
        await scope.list_shared_with_me(mode="local")

    assert policy.calls == []
