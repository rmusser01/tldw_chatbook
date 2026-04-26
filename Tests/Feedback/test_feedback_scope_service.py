import pytest

from tldw_chatbook.Feedback_Interop.feedback_scope_service import FeedbackScopeService
from tldw_chatbook.Feedback_Interop.local_feedback_service import LocalFeedbackService
from tldw_chatbook.runtime_policy import PolicyDeniedError


class FakeFeedbackService:
    def __init__(self):
        self.calls = []

    async def submit_feedback(self, **kwargs):
        self.calls.append(("submit_feedback", kwargs))
        return {"ok": True, "feedback_id": "fb-1"}

    async def list_feedback(self, conversation_id):
        self.calls.append(("list_feedback", conversation_id))
        return {"ok": True, "feedback": [{"id": "fb-1", "conversation_id": conversation_id}]}

    async def get_feedback(self, feedback_id):
        self.calls.append(("get_feedback", feedback_id))
        return {"id": feedback_id, "conversation_id": "conv-1"}

    async def update_feedback(self, feedback_id, **kwargs):
        self.calls.append(("update_feedback", feedback_id, kwargs))
        return {"ok": True, "feedback_id": feedback_id}

    async def delete_feedback(self, feedback_id):
        self.calls.append(("delete_feedback", feedback_id))
        return {"ok": True, "deleted": True, "feedback_id": feedback_id}


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
async def test_feedback_scope_service_routes_server_crud_and_normalizes_records():
    server = FakeFeedbackService()
    policy = FakePolicyEnforcer()
    scope = FeedbackScopeService(server_service=server, policy_enforcer=policy)

    submitted = await scope.submit_feedback(
        mode="server",
        conversation_id="conv-1",
        message_id="msg-1",
        feedback_type="helpful",
        helpful=True,
        query="Summarize",
    )
    listed = await scope.list_feedback("conv-1", mode="server")
    updated = await scope.update_feedback("fb-1", mode="server", user_notes="Needs detail")
    deleted = await scope.delete_feedback("fb-1", mode="server")

    assert submitted["record_id"] == "server:feedback:fb-1"
    assert listed["feedback"][0]["record_id"] == "server:feedback:fb-1"
    assert updated["record_id"] == "server:feedback:fb-1"
    assert deleted["record_id"] == "server:feedback:fb-1"
    assert server.calls == [
        (
            "submit_feedback",
            {
                "conversation_id": "conv-1",
                "message_id": "msg-1",
                "feedback_type": "helpful",
                "helpful": True,
                "query": "Summarize",
            },
        ),
        ("list_feedback", "conv-1"),
        ("update_feedback", "fb-1", {"user_notes": "Needs detail"}),
        ("delete_feedback", "fb-1"),
    ]
    assert policy.calls == [
        "feedback.create.server",
        "feedback.list.server",
        "feedback.update.server",
        "feedback.delete.server",
    ]


@pytest.mark.asyncio
async def test_feedback_scope_service_routes_local_crud_and_normalizes_records(tmp_path):
    local = LocalFeedbackService(store_path=tmp_path / "feedback.json")
    policy = FakePolicyEnforcer()
    scope = FeedbackScopeService(local_service=local, server_service=None, policy_enforcer=policy)

    submitted = await scope.submit_feedback(
        mode="local",
        conversation_id="local-conv-1",
        message_id="local-msg-1",
        feedback_type="helpful",
        helpful=True,
    )
    listed = await scope.list_feedback("local-conv-1", mode="local")
    detail = await scope.get_feedback("local-fb-1", mode="local")
    updated = await scope.update_feedback("local-fb-1", mode="local", user_notes="Needs citation")
    deleted = await scope.delete_feedback("local-fb-1", mode="local")

    assert submitted["record_id"] == "local:feedback:local-fb-1"
    assert listed["feedback"][0]["record_id"] == "local:feedback:local-fb-1"
    assert detail["record_id"] == "local:feedback:local-fb-1"
    assert updated["record_id"] == "local:feedback:local-fb-1"
    assert deleted["record_id"] == "local:feedback:local-fb-1"
    assert policy.calls == [
        "feedback.create.local",
        "feedback.list.local",
        "feedback.detail.local",
        "feedback.update.local",
        "feedback.delete.local",
    ]


@pytest.mark.asyncio
async def test_feedback_scope_service_honestly_rejects_missing_local_service():
    server = FakeFeedbackService()
    scope = FeedbackScopeService(local_service=None, server_service=server, policy_enforcer=FakePolicyEnforcer())

    with pytest.raises(ValueError, match="Local feedback backend is unavailable"):
        await scope.list_feedback("conv-1", mode="local")

    assert server.calls == []


@pytest.mark.asyncio
async def test_feedback_scope_service_blocks_denied_server_action_before_dispatch():
    server = FakeFeedbackService()
    scope = FeedbackScopeService(server_service=server, policy_enforcer=FakePolicyEnforcer("server_auth_required"))

    with pytest.raises(PolicyDeniedError) as exc:
        await scope.list_feedback("conv-1", mode="server")

    assert exc.value.reason_code == "server_auth_required"
    assert server.calls == []


def test_feedback_scope_service_reports_known_unsupported_capabilities():
    scope = FeedbackScopeService(local_service=None, server_service=FakeFeedbackService())

    assert scope.list_unsupported_capabilities(mode="server") == [
        {
            "operation_id": "feedback.detail.server",
            "source": "server",
            "supported": False,
            "reason_code": "server_contract_missing",
            "user_message": "The current server feedback API lists by conversation but does not expose single-feedback detail.",
            "affected_action_ids": ["feedback.detail.server"],
        }
    ]
    assert scope.list_unsupported_capabilities(mode="local") == []


@pytest.mark.asyncio
async def test_feedback_scope_service_reports_server_detail_as_missing_contract():
    scope = FeedbackScopeService(server_service=FakeFeedbackService(), policy_enforcer=FakePolicyEnforcer())

    with pytest.raises(ValueError, match="single-feedback detail"):
        await scope.get_feedback("fb-1", mode="server")
