"""Tests for study-pack and study-suggestion endpoint wiring."""

from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    StudyPackCreateJobRequest,
    StudyPackJobAcceptedResponse,
    StudyPackJobStatusResponse,
    StudyPackSourceSelection,
    StudyPackSummaryResponse,
    SuggestionActionRequest,
    SuggestionActionResponse,
    SuggestionJobAcceptedResponse,
    SuggestionRefreshRequest,
    SuggestionSnapshotResponse,
    SuggestionStatusResponse,
    TLDWAPIClient,
)


def _job_payload(status: str = "queued") -> dict:
    return {
        "job": {
            "id": 41,
            "status": status,
            "domain": "study_packs",
            "queue": "study_packs",
            "job_type": "study_pack_generate",
        }
    }


def _pack_payload() -> dict:
    return {
        "id": 9,
        "workspace_id": "ws-1",
        "title": "Cell Biology Pack",
        "deck_id": 7,
        "source_bundle_json": {"items": [{"source_type": "note", "source_id": "note-1"}]},
        "generation_options_json": None,
        "status": "active",
        "superseded_by_pack_id": None,
        "created_at": "2026-04-22T00:00:00Z",
        "last_modified": "2026-04-22T00:01:00Z",
        "deleted": False,
        "client_id": "server-client",
        "version": 1,
    }


@pytest.mark.asyncio
async def test_study_pack_routes_wire_and_return_typed_models(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            _job_payload("queued"),
            {"job": _job_payload("completed")["job"], "study_pack": _pack_payload(), "error": None},
            _pack_payload(),
            _job_payload("queued"),
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    accepted = await client.create_study_pack_job(
        StudyPackCreateJobRequest(
            title="Cell Biology Pack",
            workspace_id="ws-1",
            source_items=[
                StudyPackSourceSelection(
                    source_type="note",
                    source_id="note-1",
                    label="Mitochondria notes",
                )
            ],
        )
    )
    status = await client.get_study_pack_job_status(41)
    pack = await client.get_study_pack(9)
    regenerated = await client.regenerate_study_pack(9)

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/flashcards/study-packs/jobs")
    assert mocked.await_args_list[0].kwargs["json_data"] == {
        "title": "Cell Biology Pack",
        "workspace_id": "ws-1",
        "deck_mode": "new",
        "source_items": [
            {
                "source_type": "note",
                "source_id": "note-1",
                "label": "Mitochondria notes",
                "locator": {},
            }
        ],
    }
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/flashcards/study-packs/jobs/41")
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/flashcards/study-packs/9")
    assert mocked.await_args_list[3].args[:2] == ("POST", "/api/v1/flashcards/study-packs/9/regenerate")
    assert isinstance(accepted, StudyPackJobAcceptedResponse)
    assert isinstance(status, StudyPackJobStatusResponse)
    assert isinstance(pack, StudyPackSummaryResponse)
    assert isinstance(regenerated, StudyPackJobAcceptedResponse)
    assert status.study_pack is not None
    assert status.study_pack.title == "Cell Biology Pack"


@pytest.mark.asyncio
async def test_study_suggestion_routes_wire_and_return_typed_models(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
                "anchor_type": "quiz_attempt",
                "anchor_id": 17,
                "status": "ready",
                "job_id": None,
                "snapshot_id": 23,
            },
            {
                "snapshot": {
                    "id": 23,
                    "service": "quiz",
                    "activity_type": "quiz_attempt",
                    "anchor_type": "quiz_attempt",
                    "anchor_id": 17,
                    "suggestion_type": "study_suggestions",
                    "status": "active",
                    "payload": {"topics": [{"id": "topic-1", "label": "ATP"}]},
                    "user_selection": None,
                    "refreshed_from_snapshot_id": None,
                    "created_at": "2026-04-22T00:00:00Z",
                    "last_modified": "2026-04-22T00:01:00Z",
                },
                "live_evidence": {"source_available": True},
            },
            {"job": {"id": 45, "status": "queued"}},
            {
                "disposition": "generated",
                "snapshot_id": 23,
                "selection_fingerprint": "fp-topic-1",
                "target_service": "flashcards",
                "target_type": "deck",
                "target_id": "7",
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    status = await client.get_study_suggestion_status("quiz_attempt", 17)
    snapshot = await client.get_study_suggestion_snapshot(23)
    refresh = await client.refresh_study_suggestion_snapshot(
        23,
        SuggestionRefreshRequest(reason="user-request"),
    )
    action = await client.trigger_study_suggestion_action(
        23,
        SuggestionActionRequest(
            target_service="flashcards",
            target_type="deck",
            action_kind="generate",
            selected_topic_ids=["topic-1"],
            has_explicit_selection=True,
        ),
    )

    assert mocked.await_args_list[0].args[:2] == (
        "GET",
        "/api/v1/study-suggestions/anchors/quiz_attempt/17/status",
    )
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/study-suggestions/snapshots/23")
    assert mocked.await_args_list[2].args[:2] == ("POST", "/api/v1/study-suggestions/snapshots/23/refresh")
    assert mocked.await_args_list[2].kwargs["json_data"] == {"reason": "user-request"}
    assert mocked.await_args_list[3].args[:2] == ("POST", "/api/v1/study-suggestions/snapshots/23/actions")
    assert mocked.await_args_list[3].kwargs["json_data"] == {
        "target_service": "flashcards",
        "target_type": "deck",
        "action_kind": "generate",
        "selected_topic_ids": ["topic-1"],
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
    assert snapshot.snapshot.payload["topics"][0]["label"] == "ATP"
