import pytest

from tldw_chatbook.Study_Interop.server_study_service import ServerStudyService
from tldw_chatbook.tldw_api import (
    FlashcardAnalyticsSummaryResponse,
    FlashcardResponse,
    StudyPackJobAcceptedResponse,
    StudyPackJobStatusResponse,
    StudyPackSummaryResponse,
    SuggestionActionResponse,
    SuggestionJobAcceptedResponse,
    SuggestionSnapshotResponse,
    SuggestionStatusResponse,
)


CARD_UUID = "00000000-0000-4000-8000-000000000001"


class FakeClient:
    def __init__(self):
        self.calls = []

    async def update_flashcard(self, card_uuid, request_data):
        self.calls.append(("update_flashcard", card_uuid, request_data.model_dump(mode="json", exclude_none=True)))
        return FlashcardResponse.model_validate(
            {
                "uuid": card_uuid,
                "deck_id": 9,
                "front": "Question",
                "back": "Answer",
                "tags": ["science"],
                "ef": 2.5,
                "interval_days": 0,
                "repetitions": 0,
                "lapses": 0,
                "queue_state": "new",
                "created_at": "2026-04-20T00:00:00Z",
                "last_modified": "2026-04-20T00:01:00Z",
                "deleted": False,
                "client_id": "server-client",
                "version": 3,
                "model_type": "basic",
                "reverse": False,
            }
        )

    async def delete_flashcard(self, card_uuid, *, expected_version):
        self.calls.append(("delete_flashcard", card_uuid, expected_version))
        return {"deleted": True}

    async def update_flashcard_deck(self, deck_id, request_data):
        self.calls.append(("update_flashcard_deck", deck_id, request_data.model_dump(mode="json", exclude_unset=True)))
        return {
            "id": deck_id,
            "name": "Biology v2",
            "description": "Updated",
            "workspace_id": None,
            "review_prompt_side": "back",
            "deleted": False,
            "client_id": "server-client",
            "version": 2,
            "scheduler_type": "fsrs",
        }

    async def get_flashcard(self, card_uuid):
        self.calls.append(("get_flashcard", card_uuid))
        return FlashcardResponse.model_validate(
            {
                "uuid": card_uuid,
                "deck_id": 9,
                "front": "Question",
                "back": "Answer",
                "tags": ["science"],
                "ef": 2.5,
                "interval_days": 0,
                "repetitions": 0,
                "lapses": 0,
                "queue_state": "new",
                "deleted": False,
                "client_id": "server-client",
                "version": 3,
                "model_type": "basic",
                "reverse": False,
            }
        )

    async def reset_flashcard_scheduling(self, card_uuid, request_data):
        self.calls.append(("reset_flashcard_scheduling", card_uuid, request_data.model_dump(mode="json")))
        return await self.get_flashcard(card_uuid)

    async def set_flashcard_tags(self, card_uuid, request_data):
        self.calls.append(("set_flashcard_tags", card_uuid, request_data.model_dump(mode="json")))
        return await self.get_flashcard(card_uuid)

    async def get_flashcard_tags(self, card_uuid):
        self.calls.append(("get_flashcard_tags", card_uuid))
        return {"items": ["science", "biology"], "count": 2}

    async def get_flashcard_analytics_summary(self, *, deck_id=None, workspace_id=None, include_workspace_items=False):
        self.calls.append(("get_flashcard_analytics_summary", deck_id, workspace_id, include_workspace_items))
        return FlashcardAnalyticsSummaryResponse.model_validate(
            {
                "reviewed_today": 3,
                "study_streak_days": 4,
                "generated_at": "2026-04-23T12:00:00Z",
                "decks": [
                    {
                        "deck_id": 9,
                        "deck_name": "Biology",
                        "total": 12,
                        "new": 4,
                        "learning": 2,
                        "due": 3,
                        "mature": 3,
                    }
                ],
            }
        )

    async def create_study_pack_job(self, request_data):
        self.calls.append(("create_study_pack_job", request_data.model_dump(mode="json", exclude_none=True)))
        return StudyPackJobAcceptedResponse.model_validate(
            {
                "job": {
                    "id": 41,
                    "status": "queued",
                    "domain": "study_packs",
                    "queue": "study_packs",
                    "job_type": "study_pack_generate",
                }
            }
        )

    async def get_study_pack_job_status(self, job_id):
        self.calls.append(("get_study_pack_job_status", job_id))
        return StudyPackJobStatusResponse.model_validate(
            {
                "job": {
                    "id": job_id,
                    "status": "completed",
                    "domain": "study_packs",
                    "queue": "study_packs",
                    "job_type": "study_pack_generate",
                },
                "study_pack": {
                    "id": 9,
                    "title": "Cell Biology Pack",
                    "deck_id": 7,
                    "status": "active",
                    "deleted": False,
                    "client_id": "server-client",
                    "version": 1,
                },
            }
        )

    async def get_study_pack(self, pack_id):
        self.calls.append(("get_study_pack", pack_id))
        return StudyPackSummaryResponse.model_validate(
            {
                "id": pack_id,
                "title": "Cell Biology Pack",
                "deck_id": 7,
                "status": "active",
                "deleted": False,
                "client_id": "server-client",
                "version": 1,
            }
        )

    async def regenerate_study_pack(self, pack_id):
        self.calls.append(("regenerate_study_pack", pack_id))
        return StudyPackJobAcceptedResponse.model_validate(
            {
                "job": {
                    "id": 42,
                    "status": "queued",
                    "domain": "study_packs",
                    "queue": "study_packs",
                    "job_type": "study_pack_generate",
                }
            }
        )

    async def get_study_suggestion_status(self, anchor_type, anchor_id):
        self.calls.append(("get_study_suggestion_status", anchor_type, anchor_id))
        return SuggestionStatusResponse.model_validate(
            {
                "anchor_type": anchor_type,
                "anchor_id": anchor_id,
                "status": "ready",
                "snapshot_id": 23,
            }
        )

    async def get_study_suggestion_snapshot(self, snapshot_id):
        self.calls.append(("get_study_suggestion_snapshot", snapshot_id))
        return SuggestionSnapshotResponse.model_validate(
            {
                "snapshot": {
                    "id": snapshot_id,
                    "service": "quiz",
                    "activity_type": "quiz_attempt",
                    "anchor_type": "quiz_attempt",
                    "anchor_id": 17,
                    "suggestion_type": "study_suggestions",
                    "status": "active",
                    "payload": {"topics": [{"id": "topic-1"}]},
                },
                "live_evidence": {"source_available": True},
            }
        )

    async def refresh_study_suggestion_snapshot(self, snapshot_id, request_data):
        self.calls.append(("refresh_study_suggestion_snapshot", snapshot_id, request_data.model_dump(mode="json", exclude_none=True)))
        return SuggestionJobAcceptedResponse.model_validate({"job": {"id": 45, "status": "queued"}})

    async def trigger_study_suggestion_action(self, snapshot_id, request_data):
        self.calls.append(("trigger_study_suggestion_action", snapshot_id, request_data.model_dump(mode="json", exclude_none=True)))
        return SuggestionActionResponse.model_validate(
            {
                "disposition": "generated",
                "snapshot_id": snapshot_id,
                "selection_fingerprint": "fp-topic-1",
                "target_service": "flashcards",
                "target_type": "deck",
                "target_id": "7",
            }
        )


@pytest.mark.asyncio
async def test_server_study_service_moves_flashcards_via_update_flashcard():
    client = FakeClient()
    service = ServerStudyService(client=client)

    moved = await service.move_flashcard(CARD_UUID, target_deck_id=9, expected_version=2)

    assert moved["uuid"] == CARD_UUID
    assert moved["deck_id"] == 9
    assert client.calls == [
        ("update_flashcard", CARD_UUID, {"deck_id": 9, "expected_version": 2}),
    ]


@pytest.mark.asyncio
async def test_server_study_service_deletes_flashcards_with_expected_version():
    client = FakeClient()
    service = ServerStudyService(client=client)

    deleted = await service.delete_flashcard("card-server-1", expected_version=2)

    assert deleted == {"deleted": True}
    assert client.calls == [("delete_flashcard", "card-server-1", 2)]


@pytest.mark.asyncio
async def test_server_study_service_rejects_missing_expected_version_for_delete():
    client = FakeClient()
    service = ServerStudyService(client=client)

    with pytest.raises(
        ValueError,
        match="expected_version is required for server flashcard deletion\\.",
    ):
        await service.delete_flashcard("card-server-1", expected_version=None)

    assert client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("expected_version", [0, -1])
async def test_server_study_service_rejects_non_positive_expected_version_for_delete(expected_version):
    client = FakeClient()
    service = ServerStudyService(client=client)

    with pytest.raises(
        ValueError,
        match="expected_version must be >= 1 for server flashcard deletion\\.",
    ):
        await service.delete_flashcard("card-server-1", expected_version=expected_version)

    assert client.calls == []


@pytest.mark.asyncio
async def test_server_deck_delete_is_explicitly_unsupported():
    server = ServerStudyService(client=FakeClient())

    with pytest.raises(
        NotImplementedError,
        match="Flashcard deck deletion is not supported by the current server API\\.",
    ):
        await server.delete_deck(deck_id=7, expected_version=2)


@pytest.mark.asyncio
async def test_server_study_service_exposes_flashcard_management_routes():
    client = FakeClient()
    service = ServerStudyService(client=client)

    deck = await service.update_deck(
        7,
        name="Biology v2",
        description="Updated",
        review_prompt_side="back",
        expected_version=1,
    )
    card = await service.get_flashcard(CARD_UUID)
    reset = await service.reset_flashcard_scheduling(CARD_UUID, expected_version=3)
    tagged = await service.set_flashcard_tags(CARD_UUID, tags=["science", "biology"])
    tags = await service.get_flashcard_tags(CARD_UUID)
    analytics = await service.get_flashcard_analytics_summary(deck_id=9, workspace_id="ws-1", include_workspace_items=True)

    assert deck["review_prompt_side"] == "back"
    assert card["uuid"] == CARD_UUID
    assert reset["uuid"] == CARD_UUID
    assert tagged["uuid"] == CARD_UUID
    assert tags == {"items": ["science", "biology"], "count": 2}
    assert analytics["decks"][0]["deck_name"] == "Biology"
    assert client.calls[:6] == [
        ("update_flashcard_deck", 7, {"name": "Biology v2", "description": "Updated", "review_prompt_side": "back", "expected_version": 1}),
        ("get_flashcard", CARD_UUID),
        ("reset_flashcard_scheduling", CARD_UUID, {"expected_version": 3}),
        ("get_flashcard", CARD_UUID),
        ("set_flashcard_tags", CARD_UUID, {"tags": ["science", "biology"]}),
        ("get_flashcard", CARD_UUID),
    ]
    assert client.calls[-2:] == [
        ("get_flashcard_tags", CARD_UUID),
        ("get_flashcard_analytics_summary", 9, "ws-1", True),
    ]


@pytest.mark.asyncio
async def test_server_study_service_exposes_study_pack_routes():
    client = FakeClient()
    service = ServerStudyService(client=client)

    created = await service.create_study_pack_job(
        title="Cell Biology Pack",
        workspace_id="ws-1",
        source_items=[{"source_type": "note", "source_id": "note-1"}],
    )
    status = await service.get_study_pack_job_status(41)
    pack = await service.get_study_pack(9)
    regenerated = await service.regenerate_study_pack(9)

    assert created["job"]["id"] == 41
    assert status["study_pack"]["title"] == "Cell Biology Pack"
    assert pack["record_id"] == "server:study-pack:9"
    assert regenerated["job"]["id"] == 42
    assert client.calls[-4:] == [
        (
            "create_study_pack_job",
            {
                "title": "Cell Biology Pack",
                "workspace_id": "ws-1",
                "deck_mode": "new",
                "source_items": [{"source_type": "note", "source_id": "note-1", "locator": {}}],
            },
        ),
        ("get_study_pack_job_status", 41),
        ("get_study_pack", 9),
        ("regenerate_study_pack", 9),
    ]


@pytest.mark.asyncio
async def test_server_study_service_exposes_study_suggestion_routes():
    client = FakeClient()
    service = ServerStudyService(client=client)

    status = await service.get_study_suggestion_status(anchor_type="quiz_attempt", anchor_id=17)
    snapshot = await service.get_study_suggestion_snapshot(23)
    refresh = await service.refresh_study_suggestion_snapshot(23, reason="user-request")
    action = await service.trigger_study_suggestion_action(
        23,
        target_service="flashcards",
        target_type="deck",
        action_kind="generate",
        selected_topic_ids=["topic-1"],
        has_explicit_selection=True,
    )

    assert status["status"] == "ready"
    assert snapshot["snapshot"]["payload"]["topics"][0]["id"] == "topic-1"
    assert refresh["job"]["id"] == 45
    assert action["target_id"] == "7"
    assert client.calls[-4:] == [
        ("get_study_suggestion_status", "quiz_attempt", 17),
        ("get_study_suggestion_snapshot", 23),
        ("refresh_study_suggestion_snapshot", 23, {"reason": "user-request"}),
        (
            "trigger_study_suggestion_action",
            23,
            {
                "target_service": "flashcards",
                "target_type": "deck",
                "action_kind": "generate",
                "selected_topic_ids": ["topic-1"],
                "selected_topic_edits": [],
                "manual_topic_labels": [],
                "has_explicit_selection": True,
                "generator_version": "v1",
                "force_regenerate": False,
            },
        ),
    ]
