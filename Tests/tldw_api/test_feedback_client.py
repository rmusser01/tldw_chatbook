from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    ExplicitFeedbackRequest,
    ExplicitFeedbackResponse,
    FeedbackDeleteResponse,
    FeedbackListResponse,
    FeedbackUpdateRequest,
    TLDWAPIClient,
)


@pytest.mark.asyncio
async def test_feedback_client_routes_submit_list_update_delete(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {"ok": True, "feedback_id": "fb-1"},
            {
                "ok": True,
                "feedback": [
                    {
                        "id": "fb-1",
                        "conversation_id": "conv-1",
                        "message_id": "msg-1",
                        "query": "Summarize this",
                        "document_ids": ["doc-1"],
                        "chunk_ids": ["chunk-1"],
                        "relevance_score": None,
                        "helpful": True,
                        "issues": [],
                        "user_notes": "Useful",
                        "created_at": "2026-04-25T12:00:00Z",
                    }
                ],
            },
            {"ok": True, "feedback_id": "fb-1"},
            {"ok": True, "deleted": True},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    submitted = await client.submit_explicit_feedback(
        ExplicitFeedbackRequest(
            conversation_id="conv-1",
            message_id="msg-1",
            feedback_type="helpful",
            helpful=True,
            query="Summarize this",
            document_ids=["doc-1"],
            chunk_ids=["chunk-1"],
            user_notes="Useful",
            idempotency_key="idem-1",
        )
    )
    listed = await client.list_feedback("conv-1")
    updated = await client.update_feedback(
        "fb-1",
        FeedbackUpdateRequest(issues=["missing_details"], user_notes="Needs more detail"),
    )
    deleted = await client.delete_feedback("fb-1")

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/feedback/explicit")
    assert mocked.await_args_list[0].kwargs["json_data"] == {
        "conversation_id": "conv-1",
        "message_id": "msg-1",
        "feedback_type": "helpful",
        "helpful": True,
        "document_ids": ["doc-1"],
        "chunk_ids": ["chunk-1"],
        "user_notes": "Useful",
        "query": "Summarize this",
        "idempotency_key": "idem-1",
    }
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/feedback")
    assert mocked.await_args_list[1].kwargs["params"] == {"conversation_id": "conv-1"}
    assert mocked.await_args_list[2].args[:2] == ("PATCH", "/api/v1/feedback/fb-1")
    assert mocked.await_args_list[2].kwargs["json_data"] == {
        "issues": ["missing_details"],
        "user_notes": "Needs more detail",
    }
    assert mocked.await_args_list[3].args[:2] == ("DELETE", "/api/v1/feedback/fb-1")

    assert isinstance(submitted, ExplicitFeedbackResponse)
    assert isinstance(listed, FeedbackListResponse)
    assert listed.feedback[0].id == "fb-1"
    assert isinstance(updated, ExplicitFeedbackResponse)
    assert isinstance(deleted, FeedbackDeleteResponse)
    assert deleted.deleted is True
