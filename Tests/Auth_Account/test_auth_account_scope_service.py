import pytest

from tldw_chatbook.Auth_Account_Interop.auth_account_scope_service import AuthAccountScopeService
from tldw_chatbook.runtime_policy import PolicyDeniedError


class FakeAuthAccountService:
    def __init__(self):
        self.calls = []

    async def login(self, **kwargs):
        self.calls.append(("login", kwargs))
        return {"access_token": "access-1", "token_type": "bearer"}

    async def list_auth_sessions(self):
        self.calls.append(("list_auth_sessions",))
        return [{"id": 7, "ip_address": "127.0.0.1"}]

    async def get_current_user_profile(self, **kwargs):
        self.calls.append(("get_current_user_profile", kwargs))
        return {"user": {"id": 1, "username": "ada"}, "preferences": {"ui.theme": "dark"}}

    async def list_user_provider_keys(self):
        self.calls.append(("list_user_provider_keys",))
        return {"items": [{"provider": "openai", "has_key": True}]}

    async def list_storage_files(self, **kwargs):
        self.calls.append(("list_storage_files", kwargs))
        return {"items": [{"id": 9, "filename": "digest.md"}], "total": 1}


class FakePolicyEnforcer:
    def __init__(self, denied_reason=None):
        self.denied_reason = denied_reason
        self.calls = []

    def require_allowed(self, *, action_id):
        self.calls.append(action_id)
        if self.denied_reason:
            raise PolicyDeniedError(
                action_id=action_id,
                reason_code=self.denied_reason,
                user_message=f"{action_id} denied",
                effective_source="server",
                authority_owner="server",
            )


@pytest.mark.asyncio
async def test_auth_account_scope_service_routes_remote_account_surfaces_and_normalizes_records():
    server = FakeAuthAccountService()
    policy = FakePolicyEnforcer()
    scope = AuthAccountScopeService(server_service=server, policy_enforcer=policy)

    login = await scope.login(mode="server", username="ada@example.com", password="secret")
    sessions = await scope.list_auth_sessions(mode="server")
    profile = await scope.get_current_user_profile(mode="server", sections=["preferences"])
    provider_keys = await scope.list_user_provider_keys(mode="server")
    storage = await scope.list_storage_files(mode="server", limit=10)

    assert login["record_id"] == "server:auth:identity"
    assert sessions[0]["record_id"] == "server:auth_session:7"
    assert profile["record_id"] == "server:auth_profile:self"
    assert provider_keys["items"][0]["record_id"] == "server:provider_key:openai"
    assert storage["items"][0]["record_id"] == "server:storage_file:9"
    assert server.calls == [
        ("login", {"username": "ada@example.com", "password": "secret"}),
        ("list_auth_sessions",),
        ("get_current_user_profile", {"sections": ["preferences"], "include_sources": False}),
        ("list_user_provider_keys",),
        (
            "list_storage_files",
            {
                "offset": 0,
                "limit": 10,
                "file_category": None,
                "source_feature": None,
                "folder_tag": None,
                "search": None,
                "include_deleted": False,
            },
        ),
    ]
    assert policy.calls == [
        "auth.identity.launch.server",
        "auth.sessions.list.server",
        "auth.profile.detail.server",
        "auth.provider_keys.list.server",
        "auth.storage.list.server",
    ]


@pytest.mark.asyncio
async def test_auth_account_scope_service_rejects_local_mode_without_local_identity_authority():
    server = FakeAuthAccountService()
    scope = AuthAccountScopeService(server_service=server)

    with pytest.raises(ValueError, match="server-owned"):
        await scope.list_auth_sessions(mode="local")

    assert server.calls == []


@pytest.mark.asyncio
async def test_auth_account_scope_service_blocks_denied_action_before_dispatch():
    server = FakeAuthAccountService()
    scope = AuthAccountScopeService(
        server_service=server,
        policy_enforcer=FakePolicyEnforcer("server_auth_required"),
    )

    with pytest.raises(PolicyDeniedError):
        await scope.list_auth_sessions(mode="server")

    assert server.calls == []


def test_auth_account_scope_service_reports_known_unsupported_capabilities():
    scope = AuthAccountScopeService(server_service=None)

    assert scope.list_unsupported_capabilities(mode="local") == [
        {
            "operation_id": "auth_account.remote_only.local",
            "source": "local",
            "supported": False,
            "reason_code": "remote_only_surface",
            "user_message": "Server auth/profile/session state is unavailable in local/offline mode; Chatbook local identity remains single-user and separate.",
            "affected_action_ids": [],
        }
    ]
    assert scope.list_unsupported_capabilities(mode="server") == [
        {
            "operation_id": "auth_account.durable_credential_storage.server",
            "source": "server",
            "supported": False,
            "reason_code": "deferred_local_storage_policy",
            "user_message": "Durable local credential storage, auto-refresh policy, and server-switch cache invalidation remain deferred; this seam only executes explicit active-server account operations.",
            "affected_action_ids": [],
        },
        {
            "operation_id": "auth_account.admin_user_management.server",
            "source": "server",
            "supported": False,
            "reason_code": "out_of_scope_admin_surface",
            "user_message": "Admin/ops user-management surfaces are intentionally outside Chatbook client parity.",
            "affected_action_ids": [],
        },
    ]
