from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    APIKeyCreateRequest,
    APIKeyCreateResponse,
    APIKeyMetadata,
    APIKeyRotateRequest,
    AuthTokenResponse,
    BulkDeleteRequest,
    BulkDeleteResponse,
    BulkMoveRequest,
    BulkMoveResponse,
    FolderListResponse,
    GeneratedFileResponse,
    GeneratedFilesListResponse,
    GeneratedFileUpdate,
    MFASetupResponse,
    MessageResponse,
    OpenAICredentialSourceSwitchRequest,
    OpenAICredentialSourceSwitchResponse,
    OpenAIOAuthAuthorizeRequest,
    OpenAIOAuthAuthorizeResponse,
    OpenAIOAuthRefreshResponse,
    OpenAIOAuthStatusResponse,
    PasswordChangeRequest,
    PasswordResetConfirm,
    PasswordResetRequest,
    ProviderKeyTestRequest,
    ProviderKeyTestResponse,
    RestoreResponse,
    StorageQuotaResponse,
    StorageUsageResponse,
    TLDWAPIClient,
    TrashListResponse,
    UserProviderKeyResponse,
    UserProviderKeysResponse,
    UserProviderKeyUpsertRequest,
    UsageBreakdownResponse,
)


def _file_payload(file_id: int = 7) -> dict:
    return {
        "id": file_id,
        "uuid": f"file-{file_id}",
        "user_id": 1,
        "filename": "digest.md",
        "original_filename": "Digest.md",
        "storage_path": "outputs/digest.md",
        "mime_type": "text/markdown",
        "file_size_bytes": 128,
        "file_category": "spreadsheet",
        "source_feature": "data_tables",
        "folder_tag": "reports",
        "tags": ["daily"],
        "is_transient": False,
        "retention_policy": "user_default",
        "is_deleted": False,
        "created_at": "2026-04-25T12:00:00Z",
        "updated_at": "2026-04-25T12:00:00Z",
    }


@pytest.mark.asyncio
async def test_account_security_routes_wire_password_magic_mfa_api_keys_and_quota(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {"message": "Password changed successfully", "details": {"user_id": 1}},
            {"message": "If the email exists, a reset link has been sent"},
            {"message": "Password has been reset successfully"},
            {"message": "Email verified successfully"},
            {"message": "If the account exists and needs verification, an email has been sent"},
            {"message": "If the account exists, a sign-in link has been sent"},
            {"access_token": "access-1", "refresh_token": "refresh-1", "token_type": "bearer", "expires_in": 1800},
            {"secret": "totp-secret", "qr_code": "data:image/png;base64,abc", "backup_codes": ["code-1"]},
            {"message": "MFA has been enabled successfully", "backup_codes": ["code-1"]},
            {"message": "MFA has been disabled"},
            {"access_token": "access-2", "refresh_token": "refresh-2", "token_type": "bearer", "expires_in": 1800},
            [
                {
                    "id": 5,
                    "key_prefix": "tldw_1234",
                    "name": "desktop",
                    "scope": ["read", "write"],
                    "status": "active",
                    "usage_count": 0,
                }
            ],
            {
                "id": 6,
                "key_prefix": "tldw_5678",
                "name": "desktop",
                "scope": ["read"],
                "status": "active",
                "key": "tldw_secret",
            },
            {
                "id": 7,
                "key_prefix": "tldw_vkey",
                "name": "temporary",
                "scope": "read",
                "status": "active",
                "key": "tldw_virtual",
            },
            {
                "id": 8,
                "key_prefix": "tldw_rotated",
                "name": "desktop",
                "scope": ["read"],
                "status": "active",
                "key": "tldw_rotated_secret",
            },
            {"message": "API key revoked"},
            {
                "user_id": 1,
                "storage_used_mb": 12.5,
                "storage_quota_mb": 5120,
                "available_mb": 5107.5,
                "usage_percentage": 0.2,
            },
            {
                "user_id": 1,
                "storage_used_mb": 12.0,
                "storage_quota_mb": 5120,
                "available_mb": 5108.0,
                "usage_percentage": 0.2,
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    changed = await client.change_password(
        PasswordChangeRequest(current_password="OldPass123!", new_password="NewPass456!")
    )
    forgot = await client.request_password_reset(PasswordResetRequest(email="ada@example.com"))
    reset = await client.reset_password(PasswordResetConfirm(token="reset-token", new_password="NewPass456!"))
    verified = await client.verify_email("verify-token")
    resent = await client.resend_verification("ada@example.com")
    magic_requested = await client.request_magic_link("ada@example.com")
    magic_token = await client.verify_magic_link("magic-token")
    mfa_setup = await client.setup_mfa()
    mfa_verified = await client.verify_mfa_setup("123456")
    mfa_disabled = await client.disable_mfa("NewPass456!")
    mfa_token = await client.complete_mfa_login(session_token="mfa-session", mfa_token="654321")
    keys = await client.list_user_api_keys()
    created_key = await client.create_user_api_key(
        APIKeyCreateRequest(name="desktop", scope=["read"], expires_in_days=30)
    )
    virtual_key = await client.create_virtual_api_key(name="temporary", allowed_paths=["/api/v1/notes/*"])
    rotated_key = await client.rotate_user_api_key(6, APIKeyRotateRequest(expires_in_days=90))
    revoked_key = await client.revoke_user_api_key(6)
    quota = await client.get_user_storage_quota()
    recalculated = await client.recalculate_user_storage_quota()

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/users/change-password")
    assert mocked.await_args_list[0].kwargs["json_data"] == {
        "current_password": "OldPass123!",
        "new_password": "NewPass456!",
    }
    assert mocked.await_args_list[1].args[:2] == ("POST", "/api/v1/auth/forgot-password")
    assert mocked.await_args_list[1].kwargs["json_data"] == {"email": "ada@example.com"}
    assert mocked.await_args_list[2].args[:2] == ("POST", "/api/v1/auth/reset-password")
    assert mocked.await_args_list[3].args[:2] == ("GET", "/api/v1/auth/verify-email")
    assert mocked.await_args_list[3].kwargs["params"] == {"token": "verify-token"}
    assert mocked.await_args_list[4].args[:2] == ("POST", "/api/v1/auth/resend-verification")
    assert mocked.await_args_list[5].args[:2] == ("POST", "/api/v1/auth/magic-link/request")
    assert mocked.await_args_list[6].args[:2] == ("POST", "/api/v1/auth/magic-link/verify")
    assert mocked.await_args_list[7].args[:2] == ("POST", "/api/v1/auth/mfa/setup")
    assert mocked.await_args_list[8].args[:2] == ("POST", "/api/v1/auth/mfa/verify")
    assert mocked.await_args_list[8].kwargs["json_data"] == {"token": "123456"}
    assert mocked.await_args_list[9].args[:2] == ("POST", "/api/v1/auth/mfa/disable")
    assert mocked.await_args_list[9].kwargs["data"] == {"password": "NewPass456!"}
    assert mocked.await_args_list[10].args[:2] == ("POST", "/api/v1/auth/mfa/login")
    assert mocked.await_args_list[10].kwargs["json_data"] == {
        "session_token": "mfa-session",
        "mfa_token": "654321",
    }
    assert mocked.await_args_list[11].args[:2] == ("GET", "/api/v1/users/api-keys")
    assert mocked.await_args_list[12].args[:2] == ("POST", "/api/v1/users/api-keys")
    assert mocked.await_args_list[13].args[:2] == ("POST", "/api/v1/users/api-keys/virtual")
    assert mocked.await_args_list[14].args[:2] == ("POST", "/api/v1/users/api-keys/6/rotate")
    assert mocked.await_args_list[15].args[:2] == ("DELETE", "/api/v1/users/api-keys/6")
    assert mocked.await_args_list[16].args[:2] == ("GET", "/api/v1/users/storage")
    assert mocked.await_args_list[17].args[:2] == ("POST", "/api/v1/users/storage/recalculate")
    assert isinstance(changed, MessageResponse)
    assert isinstance(forgot, MessageResponse)
    assert isinstance(reset, MessageResponse)
    assert isinstance(verified, MessageResponse)
    assert isinstance(resent, MessageResponse)
    assert isinstance(magic_requested, MessageResponse)
    assert isinstance(magic_token, AuthTokenResponse)
    assert isinstance(mfa_setup, MFASetupResponse)
    assert isinstance(mfa_verified, MessageResponse)
    assert mfa_verified.details == {"backup_codes": ["code-1"]}
    assert isinstance(mfa_disabled, MessageResponse)
    assert isinstance(mfa_token, AuthTokenResponse)
    assert isinstance(keys[0], APIKeyMetadata)
    assert isinstance(created_key, APIKeyCreateResponse)
    assert isinstance(virtual_key, APIKeyCreateResponse)
    assert isinstance(rotated_key, APIKeyCreateResponse)
    assert isinstance(revoked_key, MessageResponse)
    assert isinstance(quota, StorageQuotaResponse)
    assert isinstance(recalculated, StorageQuotaResponse)


@pytest.mark.asyncio
async def test_user_provider_key_routes_wire_byok_and_openai_oauth_controls(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {"provider": "openai", "status": "stored", "key_hint": "sk-...1234", "updated_at": "2026-04-25T12:00:00Z"},
            {
                "items": [
                    {
                        "provider": "openai",
                        "has_key": True,
                        "source": "user",
                        "key_hint": "sk-...1234",
                        "auth_source": "api_key",
                    }
                ]
            },
            {"provider": "openai", "status": "valid", "model": "gpt-4o-mini"},
            {
                "provider": "openai",
                "auth_url": "https://auth.example.test/oauth",
                "auth_session_id": "session-1",
                "expires_at": "2026-04-25T12:10:00Z",
            },
            {"provider": "openai", "connected": True, "auth_source": "oauth", "scope": "model.request"},
            {"provider": "openai", "status": "refreshed", "updated_at": "2026-04-25T12:05:00Z"},
            {"provider": "openai", "auth_source": "api_key", "updated_at": "2026-04-25T12:06:00Z"},
            {},
            {},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    stored = await client.upsert_user_provider_key(
        UserProviderKeyUpsertRequest(
            provider="openai",
            api_key="sk-test",
            credential_fields={"project_id": "proj_123"},
            metadata={"label": "desktop"},
        )
    )
    listed = await client.list_user_provider_keys()
    tested = await client.test_user_provider_key(ProviderKeyTestRequest(provider="openai", model="gpt-4o-mini"))
    authz = await client.authorize_openai_oauth(
        OpenAIOAuthAuthorizeRequest(credential_fields={"project_id": "proj_123"}, return_path="/settings")
    )
    status = await client.get_openai_oauth_status()
    refreshed = await client.refresh_openai_oauth()
    switched = await client.switch_openai_credential_source(
        OpenAICredentialSourceSwitchRequest(auth_source="api_key")
    )
    disconnected = await client.disconnect_openai_oauth()
    deleted = await client.delete_user_provider_key("openai")

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/users/keys")
    assert mocked.await_args_list[0].kwargs["json_data"] == {
        "provider": "openai",
        "api_key": "sk-test",
        "credential_fields": {"project_id": "proj_123"},
        "metadata": {"label": "desktop"},
    }
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/users/keys")
    assert mocked.await_args_list[2].args[:2] == ("POST", "/api/v1/users/keys/test")
    assert mocked.await_args_list[3].args[:2] == ("POST", "/api/v1/users/keys/openai/oauth/authorize")
    assert mocked.await_args_list[4].args[:2] == ("GET", "/api/v1/users/keys/openai/oauth/status")
    assert mocked.await_args_list[5].args[:2] == ("POST", "/api/v1/users/keys/openai/oauth/refresh")
    assert mocked.await_args_list[6].args[:2] == ("POST", "/api/v1/users/keys/openai/source")
    assert mocked.await_args_list[7].args[:2] == ("DELETE", "/api/v1/users/keys/openai/oauth")
    assert mocked.await_args_list[8].args[:2] == ("DELETE", "/api/v1/users/keys/openai")
    assert isinstance(stored, UserProviderKeyResponse)
    assert isinstance(listed, UserProviderKeysResponse)
    assert isinstance(tested, ProviderKeyTestResponse)
    assert isinstance(authz, OpenAIOAuthAuthorizeResponse)
    assert isinstance(status, OpenAIOAuthStatusResponse)
    assert isinstance(refreshed, OpenAIOAuthRefreshResponse)
    assert isinstance(switched, OpenAICredentialSourceSwitchResponse)
    assert disconnected is True
    assert deleted is True


@pytest.mark.asyncio
async def test_non_admin_storage_routes_wire_file_folder_usage_and_trash(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    file_payload = _file_payload()
    mocked = AsyncMock(
        side_effect=[
            {"files": [file_payload], "total": 1, "offset": 0, "limit": 50},
            {"file": file_payload},
            {"file": {**file_payload, "folder_tag": "archive", "tags": ["daily", "archive"]}},
            {"success": True, "file_id": 7, "hard_delete": False},
            {"deleted_count": 2, "file_ids": [7, 8]},
            {"moved_count": 2, "file_ids": [7, 8], "folder_tag": "archive"},
            {"folders": [{"folder_tag": "archive", "file_count": 2, "total_bytes": 256, "total_mb": 0.0}]},
            {"success": True, "folder_tag": "archive", "message": "Folder created (virtual)"},
            {"files": [file_payload], "total": 1, "offset": 0, "limit": 20},
            {
                "usage": {
                    "total_bytes": 128,
                    "total_mb": 0.1,
                    "by_category": {"spreadsheet": {"file_count": 1, "total_bytes": 128, "total_mb": 0.1}},
                    "trash_bytes": 0,
                    "trash_mb": 0.0,
                },
                "quota_mb": 5120,
                "quota_used_mb": 0.1,
                "available_mb": 5119.9,
                "usage_percentage": 0.1,
                "at_soft_limit": False,
                "at_hard_limit": False,
            },
            {
                "user_id": 1,
                "by_category": {"spreadsheet": {"file_count": 1, "total_bytes": 128, "total_mb": 0.1}},
                "by_folder": [{"folder_tag": "archive", "file_count": 1, "total_bytes": 128, "total_mb": 0.1}],
                "total_bytes": 128,
                "total_mb": 0.1,
                "quota_mb": 5120,
                "available_mb": 5119.9,
                "usage_percentage": 0.1,
            },
            {"files": [{**file_payload, "is_deleted": True}], "total": 1, "offset": 0, "limit": 50},
            {"success": True, "file": file_payload},
            {"success": True, "file_id": 7},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    files = await client.list_storage_files(file_category="spreadsheet", source_feature="data_tables")
    one_file = await client.get_storage_file(7)
    updated = await client.update_storage_file(
        7,
        GeneratedFileUpdate(folder_tag="archive", tags=["daily", "archive"]),
    )
    deleted = await client.delete_storage_file(7)
    bulk_deleted = await client.bulk_delete_storage_files(BulkDeleteRequest(file_ids=[7, 8]))
    bulk_moved = await client.bulk_move_storage_files(BulkMoveRequest(file_ids=[7, 8], folder_tag="archive"))
    folders = await client.list_storage_folders()
    created_folder = await client.create_storage_folder("archive")
    least_accessed = await client.list_least_accessed_storage_files(limit=20)
    usage = await client.get_storage_usage()
    breakdown = await client.get_storage_usage_breakdown()
    trash = await client.list_storage_trash()
    restored = await client.restore_storage_file(7)
    permanently_deleted = await client.permanently_delete_storage_file(7)

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/storage/files")
    assert mocked.await_args_list[0].kwargs["params"] == {
        "offset": 0,
        "limit": 50,
        "file_category": "spreadsheet",
        "source_feature": "data_tables",
        "include_deleted": "false",
    }
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/storage/files/7")
    assert mocked.await_args_list[2].args[:2] == ("PATCH", "/api/v1/storage/files/7")
    assert mocked.await_args_list[2].kwargs["json_data"] == {"folder_tag": "archive", "tags": ["daily", "archive"]}
    assert mocked.await_args_list[3].args[:2] == ("DELETE", "/api/v1/storage/files/7")
    assert mocked.await_args_list[3].kwargs["params"] == {"hard_delete": "false"}
    assert mocked.await_args_list[4].args[:2] == ("POST", "/api/v1/storage/files/bulk-delete")
    assert mocked.await_args_list[5].args[:2] == ("POST", "/api/v1/storage/files/bulk-move")
    assert mocked.await_args_list[6].args[:2] == ("GET", "/api/v1/storage/folders")
    assert mocked.await_args_list[7].args[:2] == ("POST", "/api/v1/storage/folders")
    assert mocked.await_args_list[8].args[:2] == ("GET", "/api/v1/storage/files/least-accessed")
    assert mocked.await_args_list[9].args[:2] == ("GET", "/api/v1/storage/usage")
    assert mocked.await_args_list[10].args[:2] == ("GET", "/api/v1/storage/usage/breakdown")
    assert mocked.await_args_list[11].args[:2] == ("GET", "/api/v1/storage/trash")
    assert mocked.await_args_list[12].args[:2] == ("POST", "/api/v1/storage/trash/restore/7")
    assert mocked.await_args_list[13].args[:2] == ("DELETE", "/api/v1/storage/trash/7")
    assert isinstance(files, GeneratedFilesListResponse)
    assert isinstance(one_file, GeneratedFileResponse)
    assert isinstance(updated, GeneratedFileResponse)
    assert deleted["success"] is True
    assert isinstance(bulk_deleted, BulkDeleteResponse)
    assert isinstance(bulk_moved, BulkMoveResponse)
    assert isinstance(folders, FolderListResponse)
    assert created_folder["folder_tag"] == "archive"
    assert isinstance(least_accessed, GeneratedFilesListResponse)
    assert isinstance(usage, StorageUsageResponse)
    assert isinstance(breakdown, UsageBreakdownResponse)
    assert isinstance(trash, TrashListResponse)
    assert isinstance(restored, RestoreResponse)
    assert permanently_deleted["success"] is True
