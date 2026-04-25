import pytest

from tldw_chatbook.Feedback_Interop.feedback_scope_service import FeedbackScopeService
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
async def test_feedback_scope_service_honestly_rejects_local_mode_as_remote_only():
    server = FakeFeedbackService()
    scope = FeedbackScopeService(server_service=server, policy_enforcer=FakePolicyEnforcer())

    with pytest.raises(ValueError, match="Explicit feedback is server-only"):
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
    scope = FeedbackScopeService(server_service=FakeFeedbackService())

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
    assert scope.list_unsupported_capabilities(mode="local") == [
        {
            "operation_id": "feedback.remote_only.local",
            "source": "local",
            "supported": False,
            "reason_code": "remote_only_surface",
            "user_message": "Explicit feedback is unavailable in local/offline mode.",
            "affected_action_ids": [],
        }
    ]
