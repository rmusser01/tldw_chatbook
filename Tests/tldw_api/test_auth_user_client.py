from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    AuthTokenResponse,
    MessageResponse,
    RefreshTokenRequest,
    RegisterRequest,
    RegistrationResponse,
    SessionResponse,
    TLDWAPIClient,
    UserProfileCatalogResponse,
    UserProfileResponse,
    UserProfileUpdateEntry,
    UserProfileUpdateRequest,
    UserProfileUpdateResponse,
)


@pytest.mark.asyncio
async def test_auth_and_self_profile_routes_wire_login_sessions_and_profile(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {"access_token": "access-1", "refresh_token": "refresh-1", "token_type": "bearer", "expires_in": 1800},
            {"access_token": "access-2", "refresh_token": "refresh-2", "token_type": "bearer", "expires_in": 1800},
            {"message": "Successfully logged out", "details": {"user_id": 1}},
            [
                {
                    "id": 7,
                    "ip_address": "127.0.0.1",
                    "user_agent": "Chatbook",
                    "created_at": "2026-04-25T12:00:00Z",
                    "last_activity": "2026-04-25T12:01:00Z",
                    "expires_at": "2026-04-25T13:00:00Z",
                }
            ],
            {"message": "Session revoked successfully", "details": {"session_id": 7}},
            {"message": "Successfully revoked 1 sessions", "details": {"sessions_revoked": 1}},
            {
                "version": "2026-04-25",
                "updated_at": "2026-04-25T12:00:00Z",
                "entries": [{"key": "ui.theme", "label": "Theme", "type": "string", "sensitivity": "public"}],
            },
            {
                "profile_version": "2026-04-25T12:00:00Z",
                "catalog_version": "2026-04-25",
                "user": {
                    "id": 1,
                    "username": "ada",
                    "email": "ada@example.com",
                    "role": "user",
                    "is_active": True,
                    "is_verified": True,
                    "created_at": "2026-04-25T12:00:00Z",
                },
                "preferences": {"ui.theme": "light"},
            },
            {"profile_version": "2026-04-25T12:05:00Z", "applied": ["ui.theme"], "skipped": []},
            {
                "message": "Registration successful",
                "user_id": 2,
                "username": "grace",
                "email": "grace@example.com",
                "requires_verification": False,
                "api_key": "tldw_test",
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    token = await client.login("ada@example.com", "secret-password")
    refreshed = await client.refresh_auth_token(RefreshTokenRequest(refresh_token="refresh-1"))
    logout = await client.logout(all_devices=False)
    sessions = await client.list_auth_sessions()
    revoked = await client.revoke_auth_session(7)
    revoked_all = await client.revoke_all_auth_sessions()
    catalog = await client.get_user_profile_catalog(if_none_match="etag-1")
    profile = await client.get_current_user_profile(sections=["identity", "preferences"], include_sources=True)
    updated = await client.update_current_user_profile(
        UserProfileUpdateRequest(updates=[UserProfileUpdateEntry(key="ui.theme", value="dark")])
    )
    registration = await client.register_user(
        RegisterRequest(
            username="grace",
            email="grace@example.com",
            password="SecurePass123!",
        )
    )

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/auth/login")
    assert mocked.await_args_list[0].kwargs["data"] == {
        "username": "ada@example.com",
        "password": "secret-password",
        "grant_type": "password",
    }
    assert refreshed.access_token == "access-2"
    assert mocked.await_args_list[1].args[:2] == ("POST", "/api/v1/auth/refresh")
    assert mocked.await_args_list[1].kwargs["json_data"] == {"refresh_token": "refresh-1"}
    assert mocked.await_args_list[2].args[:2] == ("POST", "/api/v1/auth/logout")
    assert mocked.await_args_list[2].kwargs["json_data"] == {"all_devices": False}
    assert client.bearer_token is None
    assert mocked.await_args_list[3].args[:2] == ("GET", "/api/v1/auth/sessions")
    assert mocked.await_args_list[4].args[:2] == ("DELETE", "/api/v1/auth/sessions/7")
    assert mocked.await_args_list[5].args[:2] == ("POST", "/api/v1/auth/sessions/revoke-all")
    assert mocked.await_args_list[6].args[:2] == ("GET", "/api/v1/users/profile/catalog")
    assert mocked.await_args_list[6].kwargs["headers"] == {"If-None-Match": "etag-1"}
    assert mocked.await_args_list[7].args[:2] == ("GET", "/api/v1/users/me/profile")
    assert mocked.await_args_list[7].kwargs["params"] == {
        "sections": "identity,preferences",
        "include_sources": "true",
    }
    assert mocked.await_args_list[8].args[:2] == ("PATCH", "/api/v1/users/me/profile")
    assert mocked.await_args_list[8].kwargs["json_data"] == {
        "updates": [{"key": "ui.theme", "value": "dark"}],
        "dry_run": False,
    }
    assert mocked.await_args_list[9].args[:2] == ("POST", "/api/v1/auth/register")
    assert mocked.await_args_list[9].kwargs["json_data"] == {
        "username": "grace",
        "email": "grace@example.com",
        "password": "SecurePass123!",
    }
    assert isinstance(token, AuthTokenResponse)
    assert isinstance(refreshed, AuthTokenResponse)
    assert isinstance(logout, MessageResponse)
    assert isinstance(sessions[0], SessionResponse)
    assert revoked.details["session_id"] == 7
    assert revoked_all.details["sessions_revoked"] == 1
    assert isinstance(catalog, UserProfileCatalogResponse)
    assert isinstance(profile, UserProfileResponse)
    assert profile.preferences["ui.theme"] == "light"
    assert isinstance(updated, UserProfileUpdateResponse)
    assert isinstance(registration, RegistrationResponse)
