from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    PersonalizationDetailResponse,
    PersonalizationExplanationListResponse,
    PersonalizationMemoryCreate,
    PersonalizationMemoryExportResponse,
    PersonalizationMemoryImportRequest,
    PersonalizationMemoryItem,
    PersonalizationMemoryListResponse,
    PersonalizationMemoryUpdate,
    PersonalizationMemoryValidateRequest,
    PersonalizationOptInRequest,
    PersonalizationPreferencesUpdate,
    PersonalizationProfile,
    PersonalizationPurgeResponse,
    TLDWAPIClient,
)


PROFILE_PAYLOAD = {
    "enabled": True,
    "alpha": 0.2,
    "beta": 0.6,
    "gamma": 0.2,
    "recency_half_life_days": 14,
    "topic_count": 2,
    "memory_count": 4,
    "session_count": 1,
    "proactive_enabled": True,
    "proactive_frequency": "normal",
    "response_style": "balanced",
    "preferred_format": "auto",
    "companion_reflections_enabled": True,
    "companion_daily_reflections_enabled": False,
    "companion_weekly_reflections_enabled": True,
    "updated_at": "2026-04-22T12:00:00Z",
}


@pytest.mark.asyncio
async def test_personalization_client_wraps_profile_preference_routes(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            PROFILE_PAYLOAD,
            PROFILE_PAYLOAD | {"enabled": False},
            PROFILE_PAYLOAD | {"response_style": "concise"},
            {
                "status": "ok",
                "deleted_counts": {"semantic_memories": 4},
                "enabled": False,
                "purged_at": "2026-04-22T12:05:00Z",
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    profile = await client.get_personalization_profile()
    opted_out = await client.set_personalization_opt_in(
        PersonalizationOptInRequest(enabled=False)
    )
    updated = await client.update_personalization_preferences(
        PersonalizationPreferencesUpdate(
            response_style="concise",
            companion_daily_reflections_enabled=False,
        )
    )
    purged = await client.purge_personalization_data()

    assert isinstance(profile, PersonalizationProfile)
    assert isinstance(opted_out, PersonalizationProfile)
    assert opted_out.enabled is False
    assert isinstance(updated, PersonalizationProfile)
    assert updated.response_style == "concise"
    assert isinstance(purged, PersonalizationPurgeResponse)
    assert purged.deleted_counts == {"semantic_memories": 4}
    assert [call.args[:2] for call in mocked.await_args_list] == [
        ("GET", "/api/v1/personalization/profile"),
        ("POST", "/api/v1/personalization/opt-in"),
        ("POST", "/api/v1/personalization/preferences"),
        ("POST", "/api/v1/personalization/purge"),
    ]
    assert mocked.await_args_list[1].kwargs["json_data"] == {"enabled": False}
    assert mocked.await_args_list[2].kwargs["json_data"] == {
        "response_style": "concise",
        "companion_daily_reflections_enabled": False,
    }


@pytest.mark.asyncio
async def test_personalization_client_wraps_memory_and_explanation_routes(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    memory_payload = {
        "id": "mem-1",
        "type": "semantic",
        "content": "Prefers concise summaries",
        "pinned": True,
        "hidden": False,
        "tags": ["style"],
        "timestamp": "2026-04-22T12:00:00Z",
    }
    mocked = AsyncMock(
        side_effect=[
            {"items": [memory_payload], "total": 1, "page": 2, "size": 25},
            {"memories": [memory_payload], "total": 1},
            memory_payload,
            memory_payload | {"id": "mem-2"},
            memory_payload | {"content": "Updated"},
            {"detail": "ok: deleted"},
            {"detail": "ok: validated 1 memories"},
            {"detail": "ok: imported 1 memories"},
            {
                "items": [
                    {
                        "timestamp": "2026-04-22T12:00:00Z",
                        "context": "chat",
                        "signals": [
                            {"name": "recency", "value": 0.8, "detail": "recent"}
                        ],
                    }
                ],
                "total": 1,
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    memories = await client.list_personalization_memories(
        memory_type="semantic",
        q="concise",
        page=2,
        size=25,
        include_hidden=True,
    )
    exported = await client.export_personalization_memories()
    detail = await client.get_personalization_memory("mem-1")
    created = await client.create_personalization_memory(
        PersonalizationMemoryCreate(content="New memory", pinned=True, tags=["style"])
    )
    updated = await client.update_personalization_memory(
        "mem-1",
        PersonalizationMemoryUpdate(content="Updated"),
    )
    deleted = await client.delete_personalization_memory("mem-1")
    validated = await client.validate_personalization_memories(
        PersonalizationMemoryValidateRequest(memory_ids=["mem-1"])
    )
    imported = await client.import_personalization_memories(
        PersonalizationMemoryImportRequest(memories=[memory_payload])
    )
    explanations = await client.list_personalization_explanations(limit=5)

    assert isinstance(memories, PersonalizationMemoryListResponse)
    assert memories.items[0].id == "mem-1"
    assert isinstance(exported, PersonalizationMemoryExportResponse)
    assert isinstance(detail, PersonalizationMemoryItem)
    assert isinstance(created, PersonalizationMemoryItem)
    assert created.id == "mem-2"
    assert isinstance(updated, PersonalizationMemoryItem)
    assert updated.content == "Updated"
    assert isinstance(deleted, PersonalizationDetailResponse)
    assert isinstance(validated, PersonalizationDetailResponse)
    assert isinstance(imported, PersonalizationDetailResponse)
    assert isinstance(explanations, PersonalizationExplanationListResponse)
    assert explanations.items[0].signals[0].name == "recency"
    assert [call.args[:2] for call in mocked.await_args_list] == [
        ("GET", "/api/v1/personalization/memories"),
        ("GET", "/api/v1/personalization/memories/export"),
        ("GET", "/api/v1/personalization/memories/mem-1"),
        ("POST", "/api/v1/personalization/memories"),
        ("PATCH", "/api/v1/personalization/memories/mem-1"),
        ("DELETE", "/api/v1/personalization/memories/mem-1"),
        ("POST", "/api/v1/personalization/memories/validate"),
        ("POST", "/api/v1/personalization/memories/import"),
        ("GET", "/api/v1/personalization/explanations"),
    ]
    assert mocked.await_args_list[0].kwargs["params"] == {
        "type": "semantic",
        "q": "concise",
        "page": 2,
        "size": 25,
        "include_hidden": True,
    }
    assert mocked.await_args_list[8].kwargs["params"] == {"limit": 5}
