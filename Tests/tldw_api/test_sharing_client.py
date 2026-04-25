from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    CloneWorkspaceRequest,
    CreateTokenRequest,
    ShareWorkspaceRequest,
    SharedChatRequest,
    UpdateShareRequest,
    VerifyPasswordRequest,
    TLDWAPIClient,
)


def _share_payload() -> dict:
    return {
        "id": 7,
        "workspace_id": "ws-1",
        "owner_user_id": 1,
        "share_scope_type": "team",
        "share_scope_id": 3,
        "access_level": "view_chat",
        "allow_clone": True,
        "created_by": 1,
        "is_revoked": False,
    }


def _token_payload() -> dict:
    return {
        "id": 9,
        "token_prefix": "abc123",
        "resource_type": "workspace",
        "resource_id": "ws-1",
        "access_level": "view_chat",
        "allow_clone": True,
        "is_password_protected": False,
        "use_count": 0,
        "is_revoked": False,
        "raw_token": "secret-token",
    }


@pytest.mark.asyncio
async def test_sharing_client_routes_workspace_share_and_token_calls(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            _share_payload(),
            {"shares": [_share_payload()], "total": 1},
            {**_share_payload(), "allow_clone": False},
            {"detail": "Share revoked"},
            _token_payload(),
            {"tokens": [_token_payload()], "total": 1},
            {"detail": "Token revoked"},
            {"resource_type": "workspace", "access_level": "view_chat", "is_password_protected": True},
            {"verified": True, "session_token": "session-1"},
            {"resource_type": "workspace", "resource_id": "ws-1", "access_level": "view_chat", "owner_user_id": 1},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    share = await client.share_workspace("ws-1", ShareWorkspaceRequest(share_scope_type="team", share_scope_id=3))
    shares = await client.list_workspace_shares("ws-1", include_revoked=True)
    updated = await client.update_share(7, UpdateShareRequest(allow_clone=False))
    revoked = await client.revoke_share(7)
    token = await client.create_share_token(CreateTokenRequest(resource_type="workspace", resource_id="ws-1"))
    tokens = await client.list_share_tokens()
    token_revoked = await client.revoke_share_token(9)
    preview = await client.preview_public_share("secret-token")
    verified = await client.verify_public_share_password("secret-token", VerifyPasswordRequest(password="pass"))
    imported = await client.import_public_share("secret-token")

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/sharing/workspaces/ws-1/share")
    assert mocked.await_args_list[0].kwargs["json_data"] == {
        "share_scope_type": "team",
        "share_scope_id": 3,
        "access_level": "view_chat",
        "allow_clone": True,
    }
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/sharing/workspaces/ws-1/shares")
    assert mocked.await_args_list[1].kwargs["params"] == {"include_revoked": True}
    assert mocked.await_args_list[2].args[:2] == ("PATCH", "/api/v1/sharing/shares/7")
    assert mocked.await_args_list[2].kwargs["json_data"] == {"allow_clone": False}
    assert mocked.await_args_list[3].args[:2] == ("DELETE", "/api/v1/sharing/shares/7")
    assert mocked.await_args_list[4].args[:2] == ("POST", "/api/v1/sharing/tokens")
    assert mocked.await_args_list[4].kwargs["json_data"] == {
        "resource_type": "workspace",
        "resource_id": "ws-1",
        "access_level": "view_chat",
        "allow_clone": True,
    }
    assert mocked.await_args_list[5].args[:2] == ("GET", "/api/v1/sharing/tokens")
    assert mocked.await_args_list[6].args[:2] == ("DELETE", "/api/v1/sharing/tokens/9")
    assert mocked.await_args_list[7].args[:2] == ("GET", "/api/v1/sharing/public/secret-token")
    assert mocked.await_args_list[8].args[:2] == ("POST", "/api/v1/sharing/public/secret-token/verify")
    assert mocked.await_args_list[8].kwargs["json_data"] == {"password": "pass"}
    assert mocked.await_args_list[9].args[:2] == ("POST", "/api/v1/sharing/public/secret-token/import")

    assert share.id == 7
    assert shares.total == 1
    assert updated.allow_clone is False
    assert revoked["detail"] == "Share revoked"
    assert token.raw_token == "secret-token"
    assert tokens.total == 1
    assert token_revoked["detail"] == "Token revoked"
    assert preview.is_password_protected is True
    assert verified.session_token == "session-1"
    assert imported.resource_id == "ws-1"


@pytest.mark.asyncio
async def test_sharing_client_routes_shared_with_me_proxy_calls(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {"items": [{"share_id": 7, "workspace_id": "ws-1", "owner_user_id": 1, "access_level": "view_chat", "allow_clone": True}], "total": 1},
            {"share": _share_payload()},
            {"job_id": "job-1", "status": "pending", "message": "Clone job created"},
            [{"id": "src-1", "workspace_id": "ws-1", "title": "Source"}],
            {"id": 12, "title": "Doc", "content": "Body"},
            {"generated_answer": "Answer"},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    shared = await client.list_shared_with_me()
    workspace = await client.get_shared_workspace(7)
    clone = await client.clone_shared_workspace(7, CloneWorkspaceRequest(new_name="Clone"))
    sources = await client.list_shared_workspace_sources(7)
    media = await client.get_shared_workspace_media(7, 12)
    chat = await client.chat_with_shared_workspace(7, SharedChatRequest(query="Summarize"))

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/sharing/shared-with-me")
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/sharing/shared-with-me/7/workspace")
    assert mocked.await_args_list[2].args[:2] == ("POST", "/api/v1/sharing/shared-with-me/7/clone")
    assert mocked.await_args_list[2].kwargs["json_data"] == {"new_name": "Clone"}
    assert mocked.await_args_list[3].args[:2] == ("GET", "/api/v1/sharing/shared-with-me/7/sources")
    assert mocked.await_args_list[4].args[:2] == ("GET", "/api/v1/sharing/shared-with-me/7/media/12")
    assert mocked.await_args_list[5].args[:2] == ("POST", "/api/v1/sharing/shared-with-me/7/chat")
    assert mocked.await_args_list[5].kwargs["json_data"] == {"query": "Summarize"}

    assert shared.total == 1
    assert workspace.share.id == 7
    assert clone.job_id == "job-1"
    assert sources[0].id == "src-1"
    assert media.id == 12
    assert chat["generated_answer"] == "Answer"
