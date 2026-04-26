from __future__ import annotations

import pytest

from tldw_chatbook.Feedback_Interop.local_feedback_service import LocalFeedbackService


@pytest.mark.asyncio
async def test_local_feedback_service_persists_local_feedback_crud(tmp_path):
    service = LocalFeedbackService(store_path=tmp_path / "feedback.json")

    submitted = await service.submit_feedback(
        conversation_id="local-conv-1",
        message_id="local-msg-1",
        feedback_type="helpful",
        helpful=True,
        query="What happened?",
        user_notes="Good answer.",
    )

    assert submitted == {"ok": True, "feedback_id": "local-fb-1"}

    listed = await service.list_feedback("local-conv-1")
    assert listed["feedback"][0]["id"] == "local-fb-1"
    assert listed["feedback"][0]["conversation_id"] == "local-conv-1"

    detail = await service.get_feedback("local-fb-1")
    assert detail["id"] == "local-fb-1"
    assert detail["user_notes"] == "Good answer."

    updated = await service.update_feedback(
        "local-fb-1",
        issues=["missing_context"],
        user_notes="Needs more citation detail.",
    )
    assert updated["feedback_id"] == "local-fb-1"
    assert updated["issues"] == ["missing_context"]

    reloaded = LocalFeedbackService(store_path=tmp_path / "feedback.json")
    reloaded_detail = await reloaded.get_feedback("local-fb-1")
    assert reloaded_detail["issues"] == ["missing_context"]
    assert reloaded_detail["user_notes"] == "Needs more citation detail."

    deleted = await reloaded.delete_feedback("local-fb-1")
    assert deleted == {"ok": True, "deleted": True, "feedback_id": "local-fb-1"}
    assert await reloaded.list_feedback("local-conv-1") == {"ok": True, "feedback": []}


@pytest.mark.asyncio
async def test_local_feedback_service_uses_idempotency_key(tmp_path):
    service = LocalFeedbackService(store_path=tmp_path / "feedback.json")

    first = await service.submit_feedback(
        conversation_id="local-conv-1",
        message_id="local-msg-1",
        feedback_type="helpful",
        helpful=True,
        idempotency_key="local-key-1",
    )
    second = await service.submit_feedback(
        conversation_id="local-conv-1",
        message_id="local-msg-1",
        feedback_type="helpful",
        helpful=False,
        idempotency_key="local-key-1",
    )

    assert first == second == {"ok": True, "feedback_id": "local-fb-1"}
    assert len((await service.list_feedback("local-conv-1"))["feedback"]) == 1


@pytest.mark.asyncio
async def test_local_feedback_service_validates_feedback_requirements(tmp_path):
    service = LocalFeedbackService(store_path=tmp_path / "feedback.json")

    with pytest.raises(ValueError, match="query is required"):
        await service.submit_feedback(feedback_type="helpful", helpful=True)

    with pytest.raises(ValueError, match="helpful is required"):
        await service.submit_feedback(
            conversation_id="local-conv-1",
            message_id="local-msg-1",
            feedback_type="helpful",
        )

    with pytest.raises(ValueError, match="relevance_score is required"):
        await service.submit_feedback(
            conversation_id="local-conv-1",
            query="Find this",
            feedback_type="relevance",
        )
