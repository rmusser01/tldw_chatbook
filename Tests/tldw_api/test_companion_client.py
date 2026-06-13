from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    CompanionActivityCreate,
    CompanionActivityItem,
    CompanionActivityListResponse,
    CompanionCheckInCreate,
    CompanionConversationPromptsResponse,
    CompanionGoal,
    CompanionGoalCreate,
    CompanionGoalListResponse,
    CompanionGoalUpdate,
    CompanionKnowledgeDetail,
    CompanionKnowledgeListResponse,
    CompanionLifecycleResponse,
    CompanionPurgeRequest,
    CompanionRebuildRequest,
    CompanionReflectionDetail,
    TLDWAPIClient,
)


ACTIVITY_ITEM = {
    "id": "act-1",
    "event_type": "note.updated",
    "source_type": "note",
    "source_id": "note-1",
    "surface": "notes",
    "tags": ["research"],
    "provenance": {"note_id": "note-1"},
    "metadata": {"title": "Note"},
    "created_at": "2026-04-22T12:00:00Z",
}

GOAL_ITEM = {
    "id": "goal-1",
    "title": "Read daily",
    "description": "Read a paper every day",
    "goal_type": "habit",
    "config": {},
    "progress": {"count": 1},
    "origin_kind": "manual",
    "progress_mode": "manual",
    "derivation_key": None,
    "evidence": [],
    "status": "active",
    "created_at": "2026-04-22T12:00:00Z",
    "updated_at": "2026-04-22T12:01:00Z",
}

KNOWLEDGE_ITEM = {
    "id": "card-1",
    "card_type": "preference",
    "title": "Likes concise summaries",
    "summary": "User prefers concise summaries.",
    "evidence": [],
    "score": 0.8,
    "status": "active",
    "updated_at": "2026-04-22T12:01:00Z",
}


@pytest.mark.asyncio
async def test_companion_client_wraps_rest_routes(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            ACTIVITY_ITEM,
            ACTIVITY_ITEM | {"event_type": "manual_check_in"},
            {"items": [ACTIVITY_ITEM], "total": 1, "limit": 25, "offset": 5},
            ACTIVITY_ITEM,
            {"items": [KNOWLEDGE_ITEM], "total": 1},
            KNOWLEDGE_ITEM | {"evidence_events": [ACTIVITY_ITEM], "evidence_goals": [GOAL_ITEM]},
            {
                "id": "reflection-1",
                "title": "Daily Reflection",
                "cadence": "daily",
                "summary": "You made progress.",
                "evidence": [],
                "delivery_decision": "deliver",
                "delivery_reason": "enough_signal",
                "theme_key": "progress",
                "signal_strength": 0.7,
                "follow_up_prompts": [],
                "provenance": {},
                "created_at": "2026-04-22T12:00:00Z",
                "activity_events": [ACTIVITY_ITEM],
                "knowledge_cards": [KNOWLEDGE_ITEM],
                "goals": [GOAL_ITEM],
            },
            {
                "prompt_source_kind": "context",
                "prompt_source_id": "reflection-1",
                "prompts": [
                    {
                        "prompt_id": "prompt-1",
                        "label": "Follow up",
                        "prompt_text": "What changed?",
                        "prompt_type": "reflection",
                        "source_reflection_id": "reflection-1",
                        "source_evidence_ids": ["act-1"],
                    }
                ],
            },
            {"items": [GOAL_ITEM], "total": 1},
            GOAL_ITEM,
            GOAL_ITEM | {"title": "Read weekly"},
            {"status": "purged", "scope": "knowledge", "deleted_counts": {"cards": 1}},
            {"status": "queued", "scope": "reflections", "job_id": 42, "job_uuid": "job-uuid"},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    activity_payload = CompanionActivityCreate(
        event_type="note.updated",
        source_type="note",
        source_id="note-1",
        surface="notes",
        provenance={"note_id": "note-1"},
        tags=["research"],
    )
    check_in_payload = CompanionCheckInCreate(summary="Read a paper", tags=["habit"])
    created_activity = await client.create_companion_activity(activity_payload)
    created_check_in = await client.create_companion_check_in(check_in_payload)
    activity = await client.list_companion_activity(limit=25, offset=5)
    activity_detail = await client.get_companion_activity("act-1")
    knowledge = await client.list_companion_knowledge(status="active")
    knowledge_detail = await client.get_companion_knowledge("card-1")
    reflection_detail = await client.get_companion_reflection("reflection-1")
    prompts = await client.get_companion_conversation_prompts(query="progress")
    goals = await client.list_companion_goals(status="active")
    created_goal = await client.create_companion_goal(
        CompanionGoalCreate(title="Read daily", goal_type="habit")
    )
    updated_goal = await client.update_companion_goal(
        "goal-1",
        CompanionGoalUpdate(title="Read weekly"),
    )
    purged = await client.purge_companion_data(CompanionPurgeRequest(scope="knowledge"))
    rebuilt = await client.rebuild_companion_data(CompanionRebuildRequest(scope="reflections"))

    assert isinstance(created_activity, CompanionActivityItem)
    assert isinstance(created_check_in, CompanionActivityItem)
    assert isinstance(activity, CompanionActivityListResponse)
    assert activity.limit == 25
    assert isinstance(activity_detail, CompanionActivityItem)
    assert isinstance(knowledge, CompanionKnowledgeListResponse)
    assert isinstance(knowledge_detail, CompanionKnowledgeDetail)
    assert isinstance(reflection_detail, CompanionReflectionDetail)
    assert isinstance(prompts, CompanionConversationPromptsResponse)
    assert prompts.prompts[0].prompt_id == "prompt-1"
    assert isinstance(goals, CompanionGoalListResponse)
    assert isinstance(created_goal, CompanionGoal)
    assert isinstance(updated_goal, CompanionGoal)
    assert isinstance(purged, CompanionLifecycleResponse)
    assert isinstance(rebuilt, CompanionLifecycleResponse)

    assert [call.args[:2] for call in mocked.await_args_list] == [
        ("POST", "/api/v1/companion/activity"),
        ("POST", "/api/v1/companion/check-ins"),
        ("GET", "/api/v1/companion/activity"),
        ("GET", "/api/v1/companion/activity/act-1"),
        ("GET", "/api/v1/companion/knowledge"),
        ("GET", "/api/v1/companion/knowledge/card-1"),
        ("GET", "/api/v1/companion/reflections/reflection-1"),
        ("GET", "/api/v1/companion/conversation-prompts"),
        ("GET", "/api/v1/companion/goals"),
        ("POST", "/api/v1/companion/goals"),
        ("PATCH", "/api/v1/companion/goals/goal-1"),
        ("POST", "/api/v1/companion/purge"),
        ("POST", "/api/v1/companion/rebuild"),
    ]
    assert mocked.await_args_list[2].kwargs["params"] == {"limit": 25, "offset": 5}
    assert mocked.await_args_list[4].kwargs["params"] == {"status": "active"}
    assert mocked.await_args_list[7].kwargs["params"] == {"query": "progress"}
    assert mocked.await_args_list[8].kwargs["params"] == {"status": "active"}
