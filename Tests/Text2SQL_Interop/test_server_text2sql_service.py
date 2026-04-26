from unittest.mock import Mock

import pytest

from tldw_chatbook.Text2SQL_Interop import ServerText2SQLService
from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError


class FakeText2SQLClient:
    def __init__(self):
        self.calls = []

    async def query_text2sql(self, request_data):
        self.calls.append(("query_text2sql", request_data.model_dump(mode="json")))
        return {
            "sql": "SELECT title FROM media LIMIT 2",
            "columns": ["title"],
            "rows": [{"title": "A"}, {"title": "B"}],
            "row_count": 2,
            "duration_ms": 15,
            "target_id": "media_db",
            "guardrail": {"limit_injected": True, "limit_clamped": False},
            "truncated": False,
        }


@pytest.mark.asyncio
async def test_server_text2sql_service_routes_query_with_policy_action():
    client = FakeText2SQLClient()
    policy = Mock()
    service = ServerText2SQLService(client=client, policy_enforcer=policy)

    result = await service.query(
        query="SELECT title FROM media",
        target_id="media_db",
        max_rows=2,
        timeout_ms=1500,
        include_sql=True,
    )

    assert result["rows"] == [{"title": "A"}, {"title": "B"}]
    assert client.calls == [
        (
            "query_text2sql",
            {
                "query": "SELECT title FROM media",
                "target_id": "media_db",
                "max_rows": 2,
                "timeout_ms": 1500,
                "include_sql": True,
            },
        )
    ]
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "text2sql.query.launch.server"
    ]


@pytest.mark.asyncio
async def test_server_text2sql_service_hard_stops_denied_ui_policy_decision():
    policy = Mock()
    policy.require_allowed = None
    policy.require_ui_action_allowed = Mock(
        return_value=PolicyDecision(
            allowed=False,
            reason_code="server_auth_required",
            user_message="Blocked.",
            effective_source="server",
            authority_owner="server",
        )
    )
    client = FakeText2SQLClient()
    service = ServerText2SQLService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.query(query="SELECT 1", target_id="media_db")

    assert exc.value.reason_code == "server_auth_required"
    assert client.calls == []
