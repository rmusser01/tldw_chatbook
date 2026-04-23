from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    CloneWorkspaceRequest,
    CreateTokenRequest,
    ShareWorkspaceRequest,
    SharedChatRequest,
    TLDWAPIClient,
    UpdateShareRequest,
    VerifyPasswordRequest,
)


def _share_payload() -> dict:
    return {
        "id": 7,
        "workspace_id": "ws-1",
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


def _token_payload() -> dict:
    return {
        "id": 5,
        "token_prefix": "abc123",
        "resource_type": "workspace",
        "resource_id": "ws-1",
        "access_level": "view_chat",
        "allow_clone": True,
        "is_password_protected": False,
        "max_uses": None,
        "use_count": 0,
        "expires_at": None,
        "created_at": "2026-04-23T20:00:00Z",
        "revoked_at": None,
        "is_revoked": False,
        "raw_token": "raw-token",
    }


@pytest.mark.asyncio
async def test_client_routes_workspace_share_and_shared_with_me_calls():
    client = TLDWAPIClient("http://example.test", "token")
    client._request = AsyncMock(
        side_effect=[
            _share_payload(),
            {"shares": [_share_payload()], "total": 1},
            {**_share_payload(), "access_level": "full_edit", "allow_clone": False},
            {"detail": "Share revoked"},
            {
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
            },
            {"share": _share_payload()},
            {"job_id": "clone-job-1", "status": "pending", "message": "Clone job created"},
            [
                {
                    "id": "source-1",
                    "workspace_id": "ws-1",
                    "media_id": 42,
                    "title": "Source",
                    "source_type": "media",
                    "url": "https://example.com/source",
                    "position": 1,
                    "added_at": "2026-04-23T20:00:00Z",
                }
            ],
            {
                "id": 42,
                "title": "Media",
                "url": "https://example.com/source",
                "media_type": "article",
                "content": "Body",
                "author": "Author",
                "ingestion_date": "2026-04-23T20:00:00Z",
            },
            {"result": {"answer": "Shared answer"}},
        ]
    )

    created = await client.share_workspace(
        "ws-1",
        ShareWorkspaceRequest(
            share_scope_type="team",
            share_scope_id=11,
            access_level="view_chat",
            allow_clone=True,
        ),
    )
    listed = await client.list_workspace_shares("ws-1", include_revoked=True)
    updated = await client.update_share(
        7,
        UpdateShareRequest(access_level="full_edit", allow_clone=False),
    )
    revoked = await client.revoke_share(7)
    shared_with_me = await client.list_shared_with_me()
    shared_workspace = await client.get_shared_workspace(7)
    clone = await client.clone_shared_workspace(7, CloneWorkspaceRequest(new_name="My Copy"))
    sources = await client.list_shared_workspace_sources(7)
    media = await client.get_shared_workspace_media(7, 42)
    chat = await client.chat_with_shared_workspace(7, SharedChatRequest(query="Summarize this workspace"))

    assert created.id == 7
    assert listed.total == 1
    assert updated.access_level == "full_edit"
    assert revoked == {"detail": "Share revoked"}
    assert shared_with_me.items[0].workspace_name == "Shared Workspace"
    assert shared_workspace.share.id == 7
    assert clone.job_id == "clone-job-1"
    assert sources[0].media_id == 42
    assert media.title == "Media"
    assert chat == {"result": {"answer": "Shared answer"}}
    assert [call.args for call in client._request.await_args_list] == [
        ("POST", "/api/v1/sharing/workspaces/ws-1/share"),
        ("GET", "/api/v1/sharing/workspaces/ws-1/shares"),
        ("PATCH", "/api/v1/sharing/shares/7"),
        ("DELETE", "/api/v1/sharing/shares/7"),
        ("GET", "/api/v1/sharing/shared-with-me"),
        ("GET", "/api/v1/sharing/shared-with-me/7/workspace"),
        ("POST", "/api/v1/sharing/shared-with-me/7/clone"),
        ("GET", "/api/v1/sharing/shared-with-me/7/sources"),
        ("GET", "/api/v1/sharing/shared-with-me/7/media/42"),
        ("POST", "/api/v1/sharing/shared-with-me/7/chat"),
    ]
    assert client._request.await_args_list[0].kwargs["json_data"] == {
        "share_scope_type": "team",
        "share_scope_id": 11,
        "access_level": "view_chat",
        "allow_clone": True,
    }
    assert client._request.await_args_list[1].kwargs["params"] == {"include_revoked": "true"}
    assert client._request.await_args_list[2].kwargs["json_data"] == {
        "access_level": "full_edit",
        "allow_clone": False,
    }
    assert client._request.await_args_list[6].kwargs["json_data"] == {"new_name": "My Copy"}
    assert client._request.await_args_list[9].kwargs["json_data"] == {"query": "Summarize this workspace"}


@pytest.mark.asyncio
async def test_client_routes_share_token_calls():
    client = TLDWAPIClient("http://example.test", "token")
    client._request = AsyncMock(
        side_effect=[
            _token_payload(),
            {"tokens": [_token_payload()], "total": 1},
            {"detail": "Token revoked"},
            {
                "resource_type": "workspace",
                "resource_name": "Shared Workspace",
                "resource_description": None,
                "is_password_protected": True,
                "access_level": "view_chat",
            },
            {"verified": True, "session_token": "session-1"},
            {
                "resource_type": "workspace",
                "resource_id": "ws-1",
                "access_level": "view_chat",
                "owner_user_id": 3,
                "message": "Resource access granted. Use the resource_id to interact.",
            },
        ]
    )

    created = await client.create_share_token(
        CreateTokenRequest(
            resource_type="workspace",
            resource_id="ws-1",
            access_level="view_chat",
            allow_clone=True,
            password="passphrase",
            max_uses=10,
            expires_at="2026-05-01T00:00:00Z",
        )
    )
    listed = await client.list_share_tokens()
    revoked = await client.revoke_share_token(5)
    preview = await client.preview_public_share("raw-token")
    verified = await client.verify_public_share_password("raw-token", VerifyPasswordRequest(password="passphrase"))
    imported = await client.import_public_share("raw-token")

    assert created.raw_token == "raw-token"
    assert listed.total == 1
    assert revoked == {"detail": "Token revoked"}
    assert preview.is_password_protected is True
    assert verified.session_token == "session-1"
    assert imported.resource_id == "ws-1"
    assert [call.args for call in client._request.await_args_list] == [
        ("POST", "/api/v1/sharing/tokens"),
        ("GET", "/api/v1/sharing/tokens"),
        ("DELETE", "/api/v1/sharing/tokens/5"),
        ("GET", "/api/v1/sharing/public/raw-token"),
        ("POST", "/api/v1/sharing/public/raw-token/verify"),
        ("POST", "/api/v1/sharing/public/raw-token/import"),
    ]
    assert client._request.await_args_list[0].kwargs["json_data"] == {
        "resource_type": "workspace",
        "resource_id": "ws-1",
        "access_level": "view_chat",
        "allow_clone": True,
        "password": "passphrase",
        "max_uses": 10,
        "expires_at": "2026-05-01T00:00:00Z",
    }
    assert client._request.await_args_list[4].kwargs["json_data"] == {"password": "passphrase"}
