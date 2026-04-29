import inspect
from unittest.mock import Mock

import pytest

import tldw_chatbook.Study_Interop.server_study_service as study_module
from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError
from tldw_chatbook.Study_Interop.server_study_service import ServerStudyService
from tldw_chatbook.tldw_api.flashcards_schemas import (
    FlashcardAnalyticsSummaryResponse,
    FlashcardAssetMetadata,
    FlashcardBulkUpdateResponse,
    FlashcardDeckResponse,
    FlashcardGenerateResponse,
    FlashcardListResponse,
    FlashcardResponse,
    FlashcardReviewSessionSummary,
    FlashcardTemplateListResponse,
    FlashcardTemplateResponse,
    StudyPackJobAcceptedResponse,
    StudyPackJobStatusResponse,
    StudyPackSummaryResponse,
    FlashcardTagSuggestionsResponse,
    StudyAssistantContextResponse,
    StudyAssistantRespondResponse,
    StructuredQaImportPreviewResponse,
)
from tldw_chatbook.tldw_api import ReadingExportResponse
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

    async def delete_flashcard_deck(self, deck_id, *, expected_version):
        self.calls.append(("delete_flashcard_deck", deck_id, expected_version))
        return {"deleted": True}

    async def upload_flashcard_asset(self, file):
        self.calls.append(("upload_flashcard_asset", file))
        return FlashcardAssetMetadata.model_validate(
            {
                "asset_uuid": CARD_UUID,
                "reference": f"flashcard-asset://{CARD_UUID}",
                "markdown_snippet": f"![image](flashcard-asset://{CARD_UUID})",
                "mime_type": "image/png",
                "byte_size": 7,
                "width": 1,
                "height": 1,
                "original_filename": "cell.png",
            }
        )

    async def get_flashcard_asset_content(self, asset_uuid):
        self.calls.append(("get_flashcard_asset_content", asset_uuid))
        return ReadingExportResponse(content=b"pngdata", content_type="image/png", filename="cell.png")

    async def create_flashcards_bulk(self, request_data):
        self.calls.append(("create_flashcards_bulk", [item.model_dump(mode="json", exclude_none=True) for item in request_data]))
        return FlashcardListResponse.model_validate(
            {
                "items": [
                    {
                        "uuid": CARD_UUID,
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
                        "version": 1,
                        "model_type": "basic",
                        "reverse": False,
                    }
                ],
                "count": 1,
                "total": 1,
            }
        )

    async def update_flashcards_bulk(self, request_data):
        self.calls.append(("update_flashcards_bulk", [item.model_dump(mode="json", exclude_none=True) for item in request_data]))
        return FlashcardBulkUpdateResponse.model_validate(
            {
                "results": [
                    {
                        "uuid": CARD_UUID,
                        "status": "updated",
                        "flashcard": {
                            "uuid": CARD_UUID,
                            "deck_id": 9,
                            "front": "Updated",
                            "back": "Answer",
                            "tags": ["science"],
                            "ef": 2.5,
                            "interval_days": 0,
                            "repetitions": 0,
                            "lapses": 0,
                            "queue_state": "new",
                            "created_at": "2026-04-20T00:00:00Z",
                            "last_modified": "2026-04-20T00:02:00Z",
                            "deleted": False,
                            "client_id": "server-client",
                            "version": 2,
                            "model_type": "basic",
                            "reverse": False,
                        },
                    }
                ]
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

    async def preview_structured_qa_import(self, request_data, **limits):
        self.calls.append(("preview_structured_qa_import", request_data.model_dump(mode="json"), limits))
        return StructuredQaImportPreviewResponse.model_validate(
            {
                "drafts": [
                    {
                        "front": "What powers the cell?",
                        "back": "ATP",
                        "line_start": 1,
                        "line_end": 2,
                        "tags": ["biology"],
                    }
                ],
                "errors": [],
                "detected_format": "qa_labels",
                "skipped_blocks": 0,
            }
        )

    async def import_flashcards(self, request_data, **limits):
        self.calls.append(("import_flashcards", request_data.model_dump(mode="json", exclude_none=True), limits))
        return {"imported": 1, "items": [{"uuid": CARD_UUID, "deck_id": 9}], "errors": []}

    async def import_flashcards_json(self, file, **limits):
        self.calls.append(("import_flashcards_json", file, limits))
        return {"imported": 1, "items": [{"uuid": CARD_UUID, "deck_id": 9}], "errors": []}

    async def import_flashcards_apkg(self, file, **limits):
        self.calls.append(("import_flashcards_apkg", file, limits))
        return {"imported": 1, "items": [{"uuid": CARD_UUID, "deck_id": 9}], "errors": []}

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

    async def export_flashcards(self, **request):
        self.calls.append(("export_flashcards", request))
        return ReadingExportResponse(
            content=b"Deck\tFront\tBack\nBio\tQ\tA\n",
            content_type="text/tab-separated-values",
            filename="flashcards.tsv",
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

    async def create_flashcard_template(self, request_data):
        self.calls.append(("create_flashcard_template", request_data.model_dump(mode="json", exclude_none=True)))
        return FlashcardTemplateResponse.model_validate(
            {
                "id": 3,
                "name": "Basic science",
                "model_type": "basic",
                "front_template": "{{question}}",
                "back_template": "{{answer}}",
                "deleted": False,
                "client_id": "server-client",
                "version": 1,
            }
        )

    async def list_flashcard_templates(self, *, limit=100, offset=0):
        self.calls.append(("list_flashcard_templates", limit, offset))
        return FlashcardTemplateListResponse.model_validate(
            {
                "items": [
                    {
                        "id": 3,
                        "name": "Basic science",
                        "model_type": "basic",
                        "front_template": "{{question}}",
                        "back_template": "{{answer}}",
                        "deleted": False,
                        "client_id": "server-client",
                        "version": 1,
                    }
                ],
                "count": 1,
                "total": 1,
            }
        )

    async def get_flashcard_template(self, template_id):
        self.calls.append(("get_flashcard_template", template_id))
        return FlashcardTemplateResponse.model_validate(
            {
                "id": template_id,
                "name": "Basic science",
                "model_type": "basic",
                "front_template": "{{question}}",
                "back_template": "{{answer}}",
                "deleted": False,
                "client_id": "server-client",
                "version": 1,
            }
        )

    async def update_flashcard_template(self, template_id, request_data):
        self.calls.append(("update_flashcard_template", template_id, request_data.model_dump(mode="json", exclude_none=True)))
        return FlashcardTemplateResponse.model_validate(
            {
                "id": template_id,
                "name": "Updated science",
                "model_type": "basic",
                "front_template": "{{question}}",
                "back_template": "{{answer}}",
                "deleted": False,
                "client_id": "server-client",
                "version": 2,
            }
        )

    async def delete_flashcard_template(self, template_id, *, expected_version):
        self.calls.append(("delete_flashcard_template", template_id, expected_version))
        return {"deleted": True}

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

    async def list_study_pack_jobs(self, *, status=None, limit=100):
        self.calls.append(("list_study_pack_jobs", status, limit))
        return {
            "jobs": [
                {
                    "id": 41,
                    "status": "queued",
                    "domain": "study_packs",
                    "queue": "default",
                    "job_type": "study_pack_generate",
                }
            ],
            "total": 1,
        }

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


class FakeClientProvider:
    def __init__(self, client):
        self.client = client
        self.build_calls = 0

    def build_client(self):
        self.build_calls += 1
        return self.client


class FreshClientProvider:
    def __init__(self, factory):
        self.factory = factory
        self.build_calls = 0
        self.clients = []

    def build_client(self):
        self.build_calls += 1
        client = self.factory()
        self.clients.append(client)
        return client


class ExplodingClientProvider:
    def __init__(self):
        self.build_calls = 0

    def build_client(self):
        self.build_calls += 1
        raise AssertionError("provider should not be used when direct client exists")


def test_server_study_service_module_does_not_reference_legacy_config_client_builders():
    source = inspect.getsource(study_module)

    assert "build_runtime_api_client_from_config" not in source
    assert "build_runtime_api_client(app_config" not in source


@pytest.mark.asyncio
async def test_server_study_service_direct_client_takes_precedence_over_provider():
    client = FakeClient()
    provider = ExplodingClientProvider()
    service = ServerStudyService(client=client, client_provider=provider)

    result = await service.get_flashcard(CARD_UUID)

    assert result["uuid"] == CARD_UUID
    assert provider.build_calls == 0
    assert client.calls == [("get_flashcard", CARD_UUID)]


@pytest.mark.asyncio
async def test_server_study_service_from_server_context_provider_is_lazy():
    client = FakeClient()
    provider = FakeClientProvider(client)
    service = ServerStudyService.from_server_context_provider(provider)

    assert isinstance(service, ServerStudyService)
    assert service.client is None
    assert service.client_provider is provider
    assert provider.build_calls == 0

    result = await service.get_flashcard(CARD_UUID)

    assert result["uuid"] == CARD_UUID
    assert service.client is None
    assert provider.build_calls == 1
    assert client.calls == [("get_flashcard", CARD_UUID)]


@pytest.mark.asyncio
async def test_server_study_service_re_resolves_provider_without_service_local_client_cache():
    provider = FreshClientProvider(FakeClient)
    service = ServerStudyService.from_server_context_provider(provider)

    await service.get_flashcard(CARD_UUID)
    await service.get_flashcard(CARD_UUID)

    assert service.client is None
    assert provider.build_calls == 2
    assert len(provider.clients) == 2
    assert provider.clients[0] is not provider.clients[1]
    assert provider.clients[0].calls == [("get_flashcard", CARD_UUID)]
    assert provider.clients[1].calls == [("get_flashcard", CARD_UUID)]
    for built_client in provider.clients:
        assert all(value is not built_client for value in vars(service).values())


def test_server_study_service_from_config_returns_provider_backed_service():
    service = ServerStudyService.from_config(
        {"tldw_api": {"base_url": "https://example.com", "api_key": "test-key"}}
    )

    assert isinstance(service, ServerStudyService)
    assert service.client is None
    assert service.client_provider is not None

    client = service.client_provider.build_client()

    assert service.client is None
    assert client.base_url == "https://example.com"
    assert service.client_provider.build_client() is client


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
async def test_server_study_service_wraps_flashcard_bulk_import_export_and_asset_endpoints():
    client = FakeClient()
    service = ServerStudyService(client=client)

    uploaded = await service.upload_flashcard_asset(("cell.png", b"pngdata", "image/png"))
    content = await service.get_flashcard_asset_content(CARD_UUID)
    created_bulk = await service.create_flashcards_bulk(
        [{"deck_id": 9, "front": "Question", "back": "Answer", "tags": ["science"]}]
    )
    updated_bulk = await service.update_flashcards_bulk(
        [{"uuid": CARD_UUID, "front": "Updated", "expected_version": 1}]
    )
    preview = await service.preview_structured_qa_import("Q: What powers the cell?\nA: ATP", max_lines=10)
    imported = await service.import_flashcards("Deck\tFront\tBack\nBio\tQ\tA", has_header=True)
    imported_json = await service.import_flashcards_json(("cards.json", b"[]", "application/json"), max_items=10)
    imported_apkg = await service.import_flashcards_apkg(("cards.apkg", b"apkg", "application/octet-stream"))
    exported = await service.export_flashcards(deck_id=9, format="tsv", delimiter="\t", include_header=True)

    assert uploaded["asset_uuid"] == CARD_UUID
    assert content.content == b"pngdata"
    assert created_bulk["items"][0]["uuid"] == CARD_UUID
    assert updated_bulk["results"][0]["status"] == "updated"
    assert preview["drafts"][0]["front"] == "What powers the cell?"
    assert imported["imported"] == 1
    assert imported_json["imported"] == 1
    assert imported_apkg["imported"] == 1
    assert exported.filename == "flashcards.tsv"
    assert client.calls == [
        ("upload_flashcard_asset", ("cell.png", b"pngdata", "image/png")),
        ("get_flashcard_asset_content", CARD_UUID),
        ("create_flashcards_bulk", [{"deck_id": 9, "front": "Question", "back": "Answer", "tags": ["science"], "source_ref_type": "manual"}]),
        ("update_flashcards_bulk", [{"front": "Updated", "expected_version": 1, "uuid": CARD_UUID}]),
        ("preview_structured_qa_import", {"content": "Q: What powers the cell?\nA: ATP"}, {"max_lines": 10, "max_line_length": None, "max_field_length": None}),
        ("import_flashcards", {"content": "Deck\tFront\tBack\nBio\tQ\tA", "delimiter": "\t", "has_header": True}, {"max_lines": None, "max_line_length": None, "max_field_length": None}),
        ("import_flashcards_json", ("cards.json", b"[]", "application/json"), {"max_items": 10, "max_field_length": None}),
        ("import_flashcards_apkg", ("cards.apkg", b"apkg", "application/octet-stream"), {"max_items": None, "max_field_length": None}),
        (
            "export_flashcards",
            {
                "deck_id": 9,
                "workspace_id": None,
                "include_workspace_items": None,
                "tag": None,
                "q": None,
                "format": "tsv",
                "include_reverse": None,
                "delimiter": "\t",
                "include_header": True,
                "extended_header": None,
            },
        ),
    ]


@pytest.mark.asyncio
async def test_server_study_service_wraps_flashcard_template_endpoints():
    client = FakeClient()
    service = ServerStudyService(client=client)

    created = await service.create_flashcard_template(
        name="Basic science",
        front_template="{{question}}",
        back_template="{{answer}}",
        placeholder_definitions=[
            {
                "key": "question",
                "label": "Question",
                "required": True,
                "targets": ["front_template"],
            }
        ],
    )
    listed = await service.list_flashcard_templates(limit=5, offset=1)
    fetched = await service.get_flashcard_template(3)
    updated = await service.update_flashcard_template(3, name="Updated science", expected_version=1)
    deleted = await service.delete_flashcard_template(3, expected_version=2)

    assert created["id"] == 3
    assert listed["items"][0]["id"] == 3
    assert fetched["id"] == 3
    assert updated["version"] == 2
    assert deleted == {"deleted": True}
    assert client.calls == [
        (
            "create_flashcard_template",
            {
                "name": "Basic science",
                "model_type": "basic",
                "front_template": "{{question}}",
                "back_template": "{{answer}}",
                "placeholder_definitions": [
                    {
                        "key": "question",
                        "label": "Question",
                        "required": True,
                        "targets": ["front_template"],
                    }
                ],
            },
        ),
        ("list_flashcard_templates", 5, 1),
        ("get_flashcard_template", 3),
        ("update_flashcard_template", 3, {"name": "Updated science", "expected_version": 1}),
        ("delete_flashcard_template", 3, 2),
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
async def test_server_study_service_deletes_deck_with_expected_version():
    client = FakeClient()
    server = ServerStudyService(client=client)

    deleted = await server.delete_deck(deck_id=7, expected_version=2)

    assert deleted == {"deleted": True}
    assert client.calls == [("delete_flashcard_deck", 7, 2)]


@pytest.mark.asyncio
async def test_server_study_service_wraps_study_pack_job_endpoints():
    client = FakeClient()
    service = ServerStudyService(client=client)

    created = await service.create_study_pack_job(
        title="Cell biology pack",
        workspace_id="ws-1",
        source_items=[{"source_type": "note", "source_id": "note-1", "label": "Notes"}],
    )
    listed = await service.list_study_pack_jobs(status="queued", limit=25)
    status = await service.get_study_pack_job_status(42)
    pack = await service.get_study_pack(9)
    regenerated = await service.regenerate_study_pack(9)

    assert created["job"]["id"] == 42
    assert listed["jobs"][0]["id"] == 41
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
        ("list_study_pack_jobs", "queued", 25),
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
    await service.list_study_pack_jobs(status="queued", limit=25)
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
        "study.packs.jobs.list.server",
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
