"""Tests for study-suggestion endpoint wiring on the shared TLDW API client."""

from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    SuggestionActionRequest,
    SuggestionActionResponse,
    SuggestionJobAcceptedResponse,
    SuggestionRefreshRequest,
    SuggestionSnapshotResponse,
    SuggestionStatusResponse,
    TLDWAPIClient,
)


@pytest.mark.asyncio
async def test_study_suggestion_routes_wire_and_return_typed_models(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
                "anchor_type": "deck",
                "anchor_id": 7,
                "status": "ready",
                "job_id": None,
                "snapshot_id": 11,
            },
            {
                "snapshot": {
                    "id": 11,
                    "service": "study",
                    "activity_type": "deck_review",
                    "anchor_type": "deck",
                    "anchor_id": 7,
                    "suggestion_type": "quiz",
                    "status": "ready",
                    "payload": {"topics": [{"id": "mitosis", "label": "Mitosis"}]},
                    "user_selection": None,
                    "refreshed_from_snapshot_id": None,
                    "created_at": "2026-04-21T00:00:00Z",
                    "last_modified": "2026-04-21T00:01:00Z",
                },
                "live_evidence": {"deck_id": 7},
            },
            {"job": {"id": 44, "status": "queued"}},
            {
                "disposition": "generated",
                "snapshot_id": 11,
                "selection_fingerprint": "fp-1",
                "target_service": "quiz",
                "target_type": "quiz",
                "target_id": "quiz-9",
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    status = await client.get_study_suggestion_status(anchor_type="deck", anchor_id=7)
    snapshot = await client.get_study_suggestion_snapshot(11)
    refresh = await client.refresh_study_suggestion_snapshot(
        11,
        SuggestionRefreshRequest(reason="user_requested"),
    )
    action = await client.trigger_study_suggestion_action(
        11,
        SuggestionActionRequest(
            target_service="quiz",
            target_type="quiz",
            action_kind="generate",
            selected_topic_ids=["mitosis"],
            has_explicit_selection=True,
        ),
    )

    assert mocked.await_args_list[0].args[:2] == (
        "GET",
        "/api/v1/study-suggestions/anchors/deck/7/status",
    )
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/study-suggestions/snapshots/11")
    assert mocked.await_args_list[2].args[:2] == ("POST", "/api/v1/study-suggestions/snapshots/11/refresh")
    assert mocked.await_args_list[2].kwargs["json_data"] == {"reason": "user_requested"}
    assert mocked.await_args_list[3].args[:2] == ("POST", "/api/v1/study-suggestions/snapshots/11/actions")
    assert mocked.await_args_list[3].kwargs["json_data"] == {
        "target_service": "quiz",
        "target_type": "quiz",
        "action_kind": "generate",
        "selected_topic_ids": ["mitosis"],
        "selected_topic_edits": [],
        "manual_topic_labels": [],
        "has_explicit_selection": True,
        "generator_version": "v1",
        "force_regenerate": False,
    }

    assert isinstance(status, SuggestionStatusResponse)
    assert isinstance(snapshot, SuggestionSnapshotResponse)
    assert isinstance(refresh, SuggestionJobAcceptedResponse)
    assert isinstance(action, SuggestionActionResponse)
    assert snapshot.snapshot.id == 11
    assert action.target_id == "quiz-9"
