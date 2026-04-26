import pytest

from tldw_chatbook.Text2SQL_Interop.text2sql_scope_service import Text2SQLScopeService
from tldw_chatbook.runtime_policy import PolicyDeniedError


class FakeText2SQLService:
    def __init__(self):
        self.calls = []

    async def query(self, **kwargs):
        self.calls.append(("query", kwargs))
        return {
            "sql": "SELECT title FROM media LIMIT 2",
            "columns": ["title"],
            "rows": [{"title": "A"}, {"title": "B"}],
            "row_count": 2,
            "duration_ms": 15,
            "target_id": kwargs["target_id"],
            "guardrail": {"limit_injected": True, "limit_clamped": False},
            "truncated": False,
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


def test_text2sql_scope_service_lists_static_server_target_catalog_with_policy():
    scope = Text2SQLScopeService(
        server_service=FakeText2SQLService(),
        policy_enforcer=FakePolicyEnforcer(),
    )

    targets = scope.list_targets(mode="server")

    assert targets == [
        {
            "record_id": "server:text2sql_target:media_db",
            "backend": "server",
            "target_id": "media_db",
            "display_name": "Server Media Database",
            "description": "Read-only SQL target for the authenticated user's server media database.",
            "authorization": "checked_at_query_time",
            "query_action_id": "text2sql.query.launch.server",
        }
    ]
    assert scope.policy_enforcer.calls == ["text2sql.targets.list.server"]


def test_text2sql_scope_service_rejects_local_target_catalog_as_remote_only():
    scope = Text2SQLScopeService(
        server_service=FakeText2SQLService(),
        policy_enforcer=FakePolicyEnforcer(),
    )

    with pytest.raises(ValueError, match="Text2SQL targets are server-only"):
        scope.list_targets(mode="local")


@pytest.mark.asyncio
async def test_text2sql_scope_service_routes_server_query_and_normalizes_record():
    server = FakeText2SQLService()
    policy = FakePolicyEnforcer()
    scope = Text2SQLScopeService(server_service=server, policy_enforcer=policy)

    result = await scope.query(
        mode="server",
        query="SELECT title FROM media",
        target_id="media_db",
        max_rows=2,
        timeout_ms=1500,
        include_sql=True,
    )

    assert result["backend"] == "server"
    assert result["record_id"] == "server:text2sql_result:media_db"
    assert result["rows"] == [{"title": "A"}, {"title": "B"}]
    assert server.calls == [
        (
            "query",
            {
                "query": "SELECT title FROM media",
                "target_id": "media_db",
                "max_rows": 2,
                "timeout_ms": 1500,
                "include_sql": True,
            },
        )
    ]
    assert policy.calls == ["text2sql.query.launch.server"]


@pytest.mark.asyncio
async def test_text2sql_scope_service_honestly_rejects_local_mode_as_remote_only():
    server = FakeText2SQLService()
    scope = Text2SQLScopeService(server_service=server, policy_enforcer=FakePolicyEnforcer())

    with pytest.raises(ValueError, match="Text2SQL is server-only"):
        await scope.query(mode="local", query="SELECT 1", target_id="media_db")

    assert server.calls == []


@pytest.mark.asyncio
async def test_text2sql_scope_service_blocks_denied_server_action_before_dispatch():
    server = FakeText2SQLService()
    scope = Text2SQLScopeService(server_service=server, policy_enforcer=FakePolicyEnforcer("server_auth_required"))

    with pytest.raises(PolicyDeniedError) as exc:
        await scope.query(mode="server", query="SELECT 1", target_id="media_db")

    assert exc.value.reason_code == "server_auth_required"
    assert server.calls == []


def test_text2sql_scope_service_reports_known_unsupported_capabilities():
    scope = Text2SQLScopeService(server_service=FakeText2SQLService())

    assert scope.list_unsupported_capabilities(mode="server") == []
    assert scope.list_unsupported_capabilities(mode="local") == [
        {
            "operation_id": "text2sql.remote_only.local",
            "source": "local",
            "supported": False,
            "reason_code": "remote_only_surface",
            "user_message": "Text2SQL is unavailable in local/offline mode.",
            "affected_action_ids": [],
        }
    ]
