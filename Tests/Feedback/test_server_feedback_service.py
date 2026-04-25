from unittest.mock import Mock

import pytest

from tldw_chatbook.Feedback_Interop import ServerFeedbackService
from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError


class FakeFeedbackClient:
    def __init__(self):
        self.calls = []

    async def submit_explicit_feedback(self, request_data):
        self.calls.append(("submit_explicit_feedback", request_data.model_dump(exclude_none=True, mode="json")))
        return {"ok": True, "feedback_id": "fb-1"}

    async def list_feedback(self, conversation_id):
        self.calls.append(("list_feedback", conversation_id))
        return {
            "ok": True,
            "feedback": [
                {
                    "id": "fb-1",
                    "conversation_id": conversation_id,
                    "message_id": "msg-1",
                    "query": "Summarize",
                    "helpful": True,
                }
            ],
        }

    async def update_feedback(self, feedback_id, request_data):
        self.calls.append(("update_feedback", feedback_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {"ok": True, "feedback_id": feedback_id}

    async def delete_feedback(self, feedback_id):
        self.calls.append(("delete_feedback", feedback_id))
        return {"ok": True, "deleted": True}


@pytest.mark.asyncio
async def test_server_feedback_service_routes_crud_with_policy_actions():
    client = FakeFeedbackClient()
    policy = Mock()
    service = ServerFeedbackService(client=client, policy_enforcer=policy)

    submitted = await service.submit_feedback(
        conversation_id="conv-1",
        message_id="msg-1",
        feedback_type="helpful",
        helpful=True,
        query="Summarize",
    )
    listed = await service.list_feedback("conv-1")
    updated = await service.update_feedback("fb-1", issues=["missing_details"], user_notes="Needs detail")
    deleted = await service.delete_feedback("fb-1")

    assert submitted["feedback_id"] == "fb-1"
    assert listed["feedback"][0]["id"] == "fb-1"
    assert updated["feedback_id"] == "fb-1"
    assert deleted["deleted"] is True
    assert client.calls == [
        (
            "submit_explicit_feedback",
            {
                "conversation_id": "conv-1",
                "message_id": "msg-1",
                "feedback_type": "helpful",
                "helpful": True,
                "query": "Summarize",
            },
        ),
        ("list_feedback", "conv-1"),
        ("update_feedback", "fb-1", {"issues": ["missing_details"], "user_notes": "Needs detail"}),
        ("delete_feedback", "fb-1"),
    ]
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "feedback.create.server",
        "feedback.list.server",
        "feedback.update.server",
        "feedback.delete.server",
    ]


@pytest.mark.asyncio
async def test_server_feedback_service_hard_stops_denied_ui_policy_decision():
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
    client = FakeFeedbackClient()
    service = ServerFeedbackService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.list_feedback("conv-1")

    assert exc.value.reason_code == "server_auth_required"
    assert client.calls == []
