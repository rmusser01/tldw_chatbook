import pytest

from tldw_chatbook.User_Governance_Interop.user_governance_scope_service import UserGovernanceScopeService
from tldw_chatbook.runtime_policy import PolicyDeniedError


class FakeUserGovernanceService:
    def __init__(self):
        self.calls = []

    async def get_consent_preferences(self):
        self.calls.append(("get_consent_preferences",))
        return {
            "user_id": 42,
            "consents": [
                {
                    "id": 10,
                    "user_id": 42,
                    "purpose": "analytics",
                    "granted_at": "2026-04-25T12:00:00Z",
                    "withdrawn_at": None,
                }
            ],
        }

    async def grant_consent(self, purpose):
        self.calls.append(("grant_consent", purpose))
        return {"id": 11, "user_id": 42, "purpose": purpose}

    async def withdraw_consent(self, purpose):
        self.calls.append(("withdraw_consent", purpose))
        return {"id": 10, "user_id": 42, "purpose": purpose, "withdrawn_at": "2026-04-25T12:05:00Z"}

    async def get_self_privilege_map(self, **kwargs):
        self.calls.append(("get_self_privilege_map", kwargs))
        return {
            "catalog_version": "2026-04-25",
            "generated_at": "2026-04-25T12:00:00Z",
            "items": [
                {
                    "endpoint": "/api/v1/notes",
                    "method": "GET",
                    "privilege_scope_id": "notes.read",
                    "status": "allowed",
                }
            ],
        }

    async def get_user_privilege_map(self, user_id, **kwargs):
        self.calls.append(("get_user_privilege_map", user_id, kwargs))
        return {
            "catalog_version": "2026-04-25",
            "generated_at": "2026-04-25T12:00:00Z",
            "page": 1,
            "page_size": 25,
            "total_items": 1,
            "items": [
                {
                    "user_id": user_id,
                    "user_name": "User",
                    "endpoint": "/api/v1/notes",
                    "method": "GET",
                    "privilege_scope_id": "notes.read",
                    "status": "allowed",
                }
            ],
        }


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
async def test_user_governance_scope_service_routes_server_operations_and_normalizes_records():
    server = FakeUserGovernanceService()
    policy = FakePolicyEnforcer()
    scope = UserGovernanceScopeService(server_service=server, policy_enforcer=policy)

    preferences = await scope.get_consent_preferences(mode="server")
    granted = await scope.grant_consent("personalization", mode="server")
    withdrawn = await scope.withdraw_consent("analytics", mode="server")
    self_privileges = await scope.get_self_privilege_map(mode="server", resource="notes")
    user_privileges = await scope.get_user_privilege_map("42", mode="server", page=1, page_size=25)

    assert preferences["record_id"] == "server:consent_preferences:42"
    assert preferences["consents"][0]["record_id"] == "server:consent:10"
    assert granted["record_id"] == "server:consent:11"
    assert withdrawn["record_id"] == "server:consent:10"
    assert self_privileges["record_id"] == "server:privilege_map:self"
    assert self_privileges["items"][0]["record_id"] == "server:privilege:notes.read:GET:/api/v1/notes"
    assert user_privileges["record_id"] == "server:privilege_map:42"
    assert user_privileges["items"][0]["record_id"] == "server:privilege:42:notes.read:GET:/api/v1/notes"
    assert server.calls == [
        ("get_consent_preferences",),
        ("grant_consent", "personalization"),
        ("withdraw_consent", "analytics"),
        ("get_self_privilege_map", {"resource": "notes"}),
        ("get_user_privilege_map", "42", {"page": 1, "page_size": 25}),
    ]
    assert policy.calls == [
        "user_governance.consent.list.server",
        "user_governance.consent.update.server",
        "user_governance.consent.update.server",
        "user_governance.privileges.list.server",
        "user_governance.privileges.detail.server",
    ]


@pytest.mark.asyncio
async def test_user_governance_scope_service_honestly_rejects_local_mode_as_remote_only():
    server = FakeUserGovernanceService()
    scope = UserGovernanceScopeService(server_service=server, policy_enforcer=FakePolicyEnforcer())

    with pytest.raises(ValueError, match="User governance is server-only"):
        await scope.get_consent_preferences(mode="local")

    assert server.calls == []


@pytest.mark.asyncio
async def test_user_governance_scope_service_blocks_denied_server_action_before_dispatch():
    server = FakeUserGovernanceService()
    scope = UserGovernanceScopeService(
        server_service=server,
        policy_enforcer=FakePolicyEnforcer("server_auth_required"),
    )

    with pytest.raises(PolicyDeniedError) as exc:
        await scope.get_consent_preferences(mode="server")

    assert exc.value.reason_code == "server_auth_required"
    assert server.calls == []


def test_user_governance_scope_service_reports_known_unsupported_capabilities():
    scope = UserGovernanceScopeService(server_service=FakeUserGovernanceService())

    assert scope.list_unsupported_capabilities(mode="server") == [
        {
            "operation_id": "user_governance.admin_scopes.server",
            "source": "server",
            "supported": False,
            "reason_code": "out_of_scope_admin_surface",
            "user_message": "Org/team privilege maps, privilege snapshots, exports, and resource-governor policy administration are not exposed in Chatbook.",
            "affected_action_ids": [],
        }
    ]
    assert scope.list_unsupported_capabilities(mode="local") == [
        {
            "operation_id": "user_governance.remote_only.local",
            "source": "local",
            "supported": False,
            "reason_code": "remote_only_surface",
            "user_message": "Server user-governance state is unavailable in local/offline mode.",
            "affected_action_ids": [],
        }
    ]
