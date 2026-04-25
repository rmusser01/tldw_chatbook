from unittest.mock import Mock

import pytest

from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError
from tldw_chatbook.Study_Interop.server_study_service import ServerStudyService
from tldw_chatbook.tldw_api.flashcards_schemas import (
    FlashcardAnalyticsSummaryResponse,
    FlashcardDeckResponse,
    FlashcardGenerateResponse,
    FlashcardResponse,
    FlashcardReviewSessionSummary,
    StudyPackJobAcceptedResponse,
    StudyPackJobStatusResponse,
    StudyPackSummaryResponse,
    FlashcardTagSuggestionsResponse,
    StudyAssistantContextResponse,
    StudyAssistantRespondResponse,
)
from tldw_chatbook.tldw_api.study_suggestions_schemas import (
    SuggestionActionResponse,
    SuggestionJobAcceptedResponse,
    SuggestionSnapshotResponse,
    SuggestionStatusResponse,
)


CARD_UUID = "00000000-0000-4000-8000-000000000001"


class FakeClient:
    def __init__(self):
        self.calls = []

    async def update_flashcard_deck(self, deck_id, request_data):
        self.calls.append(("update_flashcard_deck", deck_id, request_data.model_dump(mode="json", exclude_none=True)))
        return FlashcardDeckResponse.model_validate(
            {
                "id": deck_id,
                "name": "Biology Updated",
                "description": "Cells and genetics",
                "workspace_id": None,
                "created_at": "2026-04-20T00:00:00Z",
                "last_modified": "2026-04-20T00:03:00Z",
                "deleted": False,
                "client_id": "server-client",
                "version": 4,
                "scheduler_type": "fsrs",
            }
        )

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
                "created_at": "2026-04-20T00:00:00Z",
                "last_modified": "2026-04-20T00:01:00Z",
                "deleted": False,
                "client_id": "server-client",
                "version": 3,
                "model_type": "basic",
                "reverse": False,
            }
        )

    async def reset_flashcard_scheduling(self, card_uuid, request_data):
        self.calls.append(("reset_flashcard_scheduling", card_uuid, request_data.model_dump(mode="json", exclude_none=True)))
        return await self.get_flashcard(card_uuid)

    async def set_flashcard_tags(self, card_uuid, request_data):
        self.calls.append(("set_flashcard_tags", card_uuid, request_data.model_dump(mode="json", exclude_none=True)))
        return await self.get_flashcard(card_uuid)

    async def get_flashcard_tags(self, card_uuid):
        self.calls.append(("get_flashcard_tags", card_uuid))
        return {"uuid": card_uuid, "tags": ["science", "biology"]}

    async def list_flashcard_tag_suggestions(self, *, q=None, limit=50):
        self.calls.append(("list_flashcard_tag_suggestions", q, limit))
        return FlashcardTagSuggestionsResponse.model_validate(
            {"items": [{"tag": "science", "count": 12}], "count": 1}
        )

    async def get_flashcard_analytics_summary(self, *, deck_id=None, workspace_id=None, include_workspace_items=None):
        self.calls.append(("get_flashcard_analytics_summary", deck_id, workspace_id, include_workspace_items))
        return FlashcardAnalyticsSummaryResponse.model_validate(
            {
                "reviewed_today": 4,
                "study_streak_days": 3,
                "generated_at": "2026-04-20T00:04:00Z",
                "decks": [
                    {
                        "deck_id": deck_id or 9,
                        "deck_name": "Biology Updated",
                        "total": 10,
                        "new": 2,
                        "learning": 1,
                        "due": 3,
                        "mature": 4,
                    }
                ],
            }
        )

    async def list_flashcard_review_sessions(self, *, deck_id=None, scope_key=None, status=None, limit=20):
        self.calls.append(("list_flashcard_review_sessions", deck_id, scope_key, status, limit))
        return [
            FlashcardReviewSessionSummary.model_validate(
                {
                    "id": 77,
                    "deck_id": deck_id,
                    "review_mode": "deck",
                    "scope_key": scope_key or "deck:9",
                    "status": status or "active",
                    "client_id": "server-client",
                }
            )
        ]

    async def get_flashcard_assistant(self, card_uuid):
        self.calls.append(("get_flashcard_assistant", card_uuid))
        return StudyAssistantContextResponse.model_validate(
            {
                "thread": {
                    "id": 88,
                    "context_type": "flashcard",
                    "flashcard_uuid": card_uuid,
                    "message_count": 1,
                    "deleted": False,
                    "client_id": "server-client",
                    "version": 1,
                },
                "messages": [
                    {
                        "id": 1,
                        "thread_id": 88,
                        "role": "assistant",
                        "action_type": "explain",
                        "input_modality": "text",
                        "content": "Explanation",
                        "client_id": "server-client",
                    }
                ],
                "available_actions": ["explain", "follow_up"],
            }
        )

    async def respond_flashcard_assistant(self, card_uuid, request_data):
        self.calls.append(("respond_flashcard_assistant", card_uuid, request_data.model_dump(mode="json", exclude_none=True)))
        return StudyAssistantRespondResponse.model_validate(
            {
                "thread": {
                    "id": 88,
                    "context_type": "flashcard",
                    "flashcard_uuid": card_uuid,
                    "message_count": 3,
                    "deleted": False,
                    "client_id": "server-client",
                    "version": 2,
                },
                "user_message": {
                    "id": 2,
                    "thread_id": 88,
                    "role": "user",
                    "action_type": "follow_up",
                    "input_modality": "text",
                    "content": "Why?",
                    "client_id": "server-client",
                },
                "assistant_message": {
                    "id": 3,
                    "thread_id": 88,
                    "role": "assistant",
                    "action_type": "follow_up",
                    "input_modality": "text",
                    "content": "Because.",
                    "client_id": "server-client",
                },
            }
        )

    async def generate_flashcards(self, request_data):
        self.calls.append(("generate_flashcards", request_data.model_dump(mode="json", exclude_none=True)))
        return FlashcardGenerateResponse.model_validate(
            {
                "flashcards": [{"front": "Generated Q", "back": "Generated A", "tags": ["generated"]}],
                "count": 1,
            }
        )

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

    async def create_study_pack_job(self, request_data):
        self.calls.append(("create_study_pack_job", request_data.model_dump(mode="json", exclude_none=True)))
        return StudyPackJobAcceptedResponse.model_validate(
            {
                "job": {
                    "id": 42,
                    "status": "queued",
                    "domain": "study_pack",
                    "queue": "study",
                    "job_type": "generate_study_pack",
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
                    "domain": "study_pack",
                    "queue": "study",
                    "job_type": "generate_study_pack",
                },
                "study_pack": {
                    "id": 9,
                    "workspace_id": "ws-1",
                    "title": "Cell biology pack",
                    "deck_id": 7,
                    "source_bundle_json": {},
                    "generation_options_json": None,
                    "status": "active",
                    "superseded_by_pack_id": None,
                    "created_at": "2026-04-21T00:00:00Z",
                    "last_modified": "2026-04-21T00:01:00Z",
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
                "workspace_id": "ws-1",
                "title": "Cell biology pack",
                "deck_id": 7,
                "source_bundle_json": {},
                "generation_options_json": None,
                "status": "active",
                "superseded_by_pack_id": None,
                "created_at": "2026-04-21T00:00:00Z",
                "last_modified": "2026-04-21T00:01:00Z",
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
                    "id": 43,
                    "status": "queued",
                    "domain": "study_pack",
                    "queue": "study",
                    "job_type": "regenerate_study_pack",
                }
            }
        )

    async def get_study_suggestion_status(self, *, anchor_type, anchor_id):
        self.calls.append(("get_study_suggestion_status", anchor_type, anchor_id))
        return SuggestionStatusResponse.model_validate(
            {"anchor_type": anchor_type, "anchor_id": anchor_id, "status": "ready", "snapshot_id": 11}
        )

    async def get_study_suggestion_snapshot(self, snapshot_id):
        self.calls.append(("get_study_suggestion_snapshot", snapshot_id))
        return SuggestionSnapshotResponse.model_validate(
            {
                "snapshot": {
                    "id": snapshot_id,
                    "service": "study",
                    "activity_type": "deck_review",
                    "anchor_type": "deck",
                    "anchor_id": 7,
                    "suggestion_type": "quiz",
                    "status": "ready",
                    "payload": {"topics": []},
                },
                "live_evidence": {},
            }
        )

    async def refresh_study_suggestion_snapshot(self, snapshot_id, request_data):
        self.calls.append(("refresh_study_suggestion_snapshot", snapshot_id, request_data.model_dump(mode="json", exclude_none=True)))
        return SuggestionJobAcceptedResponse.model_validate({"job": {"id": 44, "status": "queued"}})

    async def trigger_study_suggestion_action(self, snapshot_id, request_data):
        self.calls.append(("trigger_study_suggestion_action", snapshot_id, request_data.model_dump(mode="json", exclude_none=True)))
        return SuggestionActionResponse.model_validate(
            {
                "disposition": "generated",
                "snapshot_id": snapshot_id,
                "selection_fingerprint": "fp-1",
                "target_service": "quiz",
                "target_type": "quiz",
                "target_id": "quiz-9",
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
async def test_server_study_service_wraps_broad_flashcard_helper_endpoints():
    client = FakeClient()
    service = ServerStudyService(client=client)

    deck = await service.update_deck(9, name="Biology Updated", description="Cells and genetics", expected_version=3)
    card = await service.get_flashcard(CARD_UUID)
    reset = await service.reset_flashcard_scheduling(CARD_UUID, expected_version=3)
    tagged = await service.set_flashcard_tags(CARD_UUID, tags=["science", "biology"])
    tags = await service.get_flashcard_tags(CARD_UUID)
    suggestions = await service.list_flashcard_tag_suggestions(q="sci", limit=5)
    analytics = await service.get_flashcard_analytics_summary(deck_id=9)
    sessions = await service.list_review_sessions(deck_id=9, status="active", limit=2)
    assistant = await service.get_flashcard_assistant(CARD_UUID)
    assistant_response = await service.respond_flashcard_assistant(
        CARD_UUID,
        action="follow_up",
        message="Why?",
        expected_thread_version=1,
    )
    generated = await service.generate_flashcards(text="Cells divide by mitosis.", num_cards=1, focus_topics=["mitosis"])

    assert deck["version"] == 4
    assert card["uuid"] == CARD_UUID
    assert reset["uuid"] == CARD_UUID
    assert tagged["uuid"] == CARD_UUID
    assert tags == {"uuid": CARD_UUID, "tags": ["science", "biology"]}
    assert suggestions["items"][0]["tag"] == "science"
    assert analytics["reviewed_today"] == 4
    assert sessions[0]["id"] == 77
    assert assistant["thread"]["id"] == 88
    assert assistant_response["assistant_message"]["content"] == "Because."
    assert generated["flashcards"][0]["front"] == "Generated Q"
    assert client.calls == [
        ("update_flashcard_deck", 9, {"name": "Biology Updated", "description": "Cells and genetics", "expected_version": 3}),
        ("get_flashcard", CARD_UUID),
        ("reset_flashcard_scheduling", CARD_UUID, {"expected_version": 3}),
        ("get_flashcard", CARD_UUID),
        ("set_flashcard_tags", CARD_UUID, {"tags": ["science", "biology"]}),
        ("get_flashcard", CARD_UUID),
        ("get_flashcard_tags", CARD_UUID),
        ("list_flashcard_tag_suggestions", "sci", 5),
        ("get_flashcard_analytics_summary", 9, None, None),
        ("list_flashcard_review_sessions", 9, None, "active", 2),
        ("get_flashcard_assistant", CARD_UUID),
        ("respond_flashcard_assistant", CARD_UUID, {"action": "follow_up", "message": "Why?", "input_modality": "text", "expected_thread_version": 1}),
        (
            "generate_flashcards",
            {
                "text": "Cells divide by mitosis.",
                "num_cards": 1,
                "card_type": "basic",
                "difficulty": "mixed",
                "focus_topics": ["mitosis"],
            },
        ),
    ]


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
async def test_server_study_service_wraps_study_pack_job_endpoints():
    client = FakeClient()
    service = ServerStudyService(client=client)

    created = await service.create_study_pack_job(
        title="Cell biology pack",
        workspace_id="ws-1",
        source_items=[{"source_type": "note", "source_id": "note-1", "label": "Notes"}],
    )
    status = await service.get_study_pack_job_status(42)
    pack = await service.get_study_pack(9)
    regenerated = await service.regenerate_study_pack(9)

    assert created["job"]["id"] == 42
    assert status["study_pack"]["id"] == 9
    assert pack["id"] == 9
    assert regenerated["job"]["id"] == 43
    assert client.calls == [
        (
            "create_study_pack_job",
            {
                "title": "Cell biology pack",
                "workspace_id": "ws-1",
                "deck_mode": "new",
                "source_items": [{"source_type": "note", "source_id": "note-1", "label": "Notes", "locator": {}}],
            },
        ),
        ("get_study_pack_job_status", 42),
        ("get_study_pack", 9),
        ("regenerate_study_pack", 9),
    ]


@pytest.mark.asyncio
async def test_server_study_service_wraps_study_suggestion_endpoints():
    client = FakeClient()
    service = ServerStudyService(client=client)

    status = await service.get_study_suggestion_status(anchor_type="deck", anchor_id=7)
    snapshot = await service.get_study_suggestion_snapshot(11)
    refresh = await service.refresh_study_suggestion_snapshot(11, reason="user_requested")
    action = await service.trigger_study_suggestion_action(
        11,
        target_service="quiz",
        target_type="quiz",
        action_kind="generate",
        selected_topic_ids=["mitosis"],
        has_explicit_selection=True,
    )

    assert status["snapshot_id"] == 11
    assert snapshot["snapshot"]["id"] == 11
    assert refresh["job"]["id"] == 44
    assert action["target_id"] == "quiz-9"
    assert client.calls == [
        ("get_study_suggestion_status", "deck", 7),
        ("get_study_suggestion_snapshot", 11),
        ("refresh_study_suggestion_snapshot", 11, {"reason": "user_requested"}),
        (
            "trigger_study_suggestion_action",
            11,
            {
                "target_service": "quiz",
                "target_type": "quiz",
                "action_kind": "generate",
                "selected_topic_ids": ["mitosis"],
                "selected_topic_edits": [],
                "manual_topic_labels": [],
                "has_explicit_selection": True,
                "generator_version": "v1",
                "force_regenerate": False,
            },
        ),
    ]


@pytest.mark.asyncio
async def test_server_study_service_enforces_remote_study_pack_and_suggestion_actions():
    client = FakeClient()
    policy = Mock()
    service = ServerStudyService(client=client, policy_enforcer=policy)

    await service.create_study_pack_job(
        title="Cell biology pack",
        workspace_id="ws-1",
        source_items=[{"source_type": "note", "source_id": "note-1", "label": "Notes"}],
    )
    await service.get_study_pack_job_status(42)
    await service.get_study_pack(9)
    await service.regenerate_study_pack(9)
    await service.get_study_suggestion_status(anchor_type="deck", anchor_id=7)
    await service.get_study_suggestion_snapshot(11)
    await service.refresh_study_suggestion_snapshot(11, reason="user_requested")
    await service.trigger_study_suggestion_action(
        11,
        target_service="quiz",
        target_type="quiz",
        action_kind="generate",
        selected_topic_ids=["mitosis"],
        has_explicit_selection=True,
    )

    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "study.packs.jobs.launch.server",
        "study.packs.jobs.observe.server",
        "study.packs.jobs.observe.server",
        "study.packs.jobs.launch.server",
        "study.suggestions.list.server",
        "study.suggestions.observe.server",
        "study.suggestions.launch.server",
        "study.suggestions.configure.server",
    ]


@pytest.mark.asyncio
async def test_server_study_service_hard_stops_denied_ui_policy_decision():
    policy = Mock()
    policy.require_allowed = None
    policy.require_ui_action_allowed = Mock(
        return_value=PolicyDecision(
            allowed=False,
            reason_code="server_unreachable",
            user_message="Blocked.",
            effective_source="server",
            authority_owner="server",
        )
    )
    client = FakeClient()
    service = ServerStudyService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.create_study_pack_job(
            title="Cell biology pack",
            source_items=[{"source_type": "note", "source_id": "note-1"}],
        )

    assert exc.value.reason_code == "server_unreachable"
    assert client.calls == []
