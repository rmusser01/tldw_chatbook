"""Tests for flashcard endpoint wiring on the shared TLDW API client."""

from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    FlashcardAnalyticsSummaryResponse,
    FlashcardAssetMetadata,
    FlashcardBulkUpdateItem,
    FlashcardBulkUpdateResponse,
    FlashcardCreateRequest,
    FlashcardDeckCreateRequest,
    FlashcardDeckResponse,
    FlashcardDeckUpdateRequest,
    FlashcardGenerateRequest,
    FlashcardGenerateResponse,
    FlashcardListResponse,
    FlashcardNextReviewResponse,
    FlashcardResponse,
    FlashcardResetSchedulingRequest,
    FlashcardReviewRequest,
    FlashcardReviewResponse,
    FlashcardReviewSessionSummary,
    FlashcardUpdateRequest,
    FlashcardTagSuggestionsResponse,
    FlashcardTagsUpdate,
    FlashcardTemplateCreateRequest,
    FlashcardTemplateListResponse,
    FlashcardTemplateResponse,
    FlashcardTemplateUpdateRequest,
    FlashcardsImportRequest,
    ReadingExportResponse,
    StructuredQaImportPreviewRequest,
    StructuredQaImportPreviewResponse,
    StudyAssistantContextResponse,
    StudyAssistantRespondRequest,
    StudyAssistantRespondResponse,
    StudyPackCreateJobRequest,
    StudyPackJobAcceptedResponse,
    StudyPackJobStatusResponse,
    StudyPackSourceSelection,
    StudyPackSummaryResponse,
    TLDWAPIClient,
)


CARD_UUID = "87ca2b3f-7e3a-47d7-a52f-8debc86c03cb"


def _deck_payload(**overrides) -> dict:
    payload = {
        "id": 7,
        "name": "Biology",
        "description": "Cell review",
        "workspace_id": None,
        "review_prompt_side": "front",
        "created_at": "2026-04-20T00:00:00Z",
        "last_modified": "2026-04-20T00:01:00Z",
        "deleted": False,
        "client_id": "server-client",
        "version": 1,
        "scheduler_type": "fsrs",
        "scheduler_settings": {"fsrs": {"target_retention": 0.9}},
    }
    payload.update(overrides)
    return payload


def _card_payload(**overrides) -> dict:
    payload = {
        "uuid": CARD_UUID,
        "deck_id": 7,
        "front": "What powers the cell?",
        "back": "ATP",
        "notes": None,
        "extra": None,
        "is_cloze": False,
        "tags": ["biology"],
        "ef": 2.5,
        "interval_days": 0,
        "repetitions": 0,
        "lapses": 0,
        "due_at": None,
        "last_reviewed_at": None,
        "queue_state": "new",
        "created_at": "2026-04-20T00:00:00Z",
        "last_modified": "2026-04-20T00:01:00Z",
        "deleted": False,
        "client_id": "server-client",
        "version": 1,
        "model_type": "basic",
        "reverse": False,
        "scheduler_type": "fsrs",
    }
    payload.update(overrides)
    return payload


def _assistant_thread(**overrides) -> dict:
    payload = {
        "id": 11,
        "context_type": "flashcard",
        "flashcard_uuid": CARD_UUID,
        "quiz_attempt_id": None,
        "question_id": None,
        "last_message_at": "2026-04-20T00:02:00Z",
        "message_count": 1,
        "deleted": False,
        "client_id": "server-client",
        "version": 1,
        "created_at": "2026-04-20T00:00:00Z",
        "last_modified": "2026-04-20T00:01:00Z",
    }
    payload.update(overrides)
    return payload


def _assistant_message(**overrides) -> dict:
    payload = {
        "id": 12,
        "thread_id": 11,
        "role": "assistant",
        "action_type": "explain",
        "input_modality": "text",
        "content": "ATP is the primary energy currency.",
        "structured_payload": {},
        "context_snapshot": {},
        "provider": "test",
        "model": "test-model",
        "created_at": "2026-04-20T00:02:00Z",
        "client_id": "server-client",
    }
    payload.update(overrides)
    return payload


def _template_payload(**overrides) -> dict:
    payload = {
        "id": 3,
        "name": "Basic science",
        "model_type": "basic",
        "front_template": "{{question}}",
        "back_template": "{{answer}}",
        "notes_template": None,
        "extra_template": None,
        "placeholder_definitions": [
            {
                "key": "question",
                "label": "Question",
                "help_text": None,
                "default_value": None,
                "required": True,
                "targets": ["front_template"],
            }
        ],
        "created_at": "2026-04-20T00:00:00Z",
        "last_modified": "2026-04-20T00:01:00Z",
        "deleted": False,
        "client_id": "server-client",
        "version": 1,
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
async def test_flashcard_routes_wire_and_return_typed_models(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
                "id": 7,
                "name": "Biology",
                "description": "Cell review",
                "workspace_id": None,
                "created_at": "2026-04-20T00:00:00Z",
                "last_modified": "2026-04-20T00:01:00Z",
                "deleted": False,
                "client_id": "server-client",
                "version": 1,
                "scheduler_type": "fsrs",
            },
            [
                {
                    "id": 7,
                    "name": "Biology",
                    "description": "Cell review",
                    "workspace_id": None,
                    "created_at": "2026-04-20T00:00:00Z",
                    "last_modified": "2026-04-20T00:01:00Z",
                    "deleted": False,
                    "client_id": "server-client",
                    "version": 1,
                    "scheduler_type": "fsrs",
                }
            ],
            {
                "uuid": "87ca2b3f-7e3a-47d7-a52f-8debc86c03cb",
                "deck_id": 7,
                "front": "What powers the cell?",
                "back": "ATP",
                "notes": None,
                "extra": None,
                "is_cloze": False,
                "tags": ["biology"],
                "ef": 2.5,
                "interval_days": 0,
                "repetitions": 0,
                "lapses": 0,
                "due_at": None,
                "last_reviewed_at": None,
                "queue_state": "new",
                "created_at": "2026-04-20T00:00:00Z",
                "last_modified": "2026-04-20T00:01:00Z",
                "deleted": False,
                "client_id": "server-client",
                "version": 1,
                "model_type": "basic",
                "reverse": False,
                "scheduler_type": "fsrs",
            },
            {
                "items": [
                    {
                        "uuid": "87ca2b3f-7e3a-47d7-a52f-8debc86c03cb",
                        "deck_id": 7,
                        "front": "What powers the cell?",
                        "back": "ATP",
                        "notes": None,
                        "extra": None,
                        "is_cloze": False,
                        "tags": ["biology"],
                        "ef": 2.5,
                        "interval_days": 0,
                        "repetitions": 0,
                        "lapses": 0,
                        "due_at": None,
                        "last_reviewed_at": None,
                        "queue_state": "new",
                        "created_at": "2026-04-20T00:00:00Z",
                        "last_modified": "2026-04-20T00:01:00Z",
                        "deleted": False,
                        "client_id": "server-client",
                        "version": 1,
                        "model_type": "basic",
                        "reverse": False,
                        "scheduler_type": "fsrs",
                    }
                ],
                "count": 1,
                "total": 1,
            },
            {
                "card": {
                    "uuid": "87ca2b3f-7e3a-47d7-a52f-8debc86c03cb",
                    "deck_id": 7,
                    "front": "What powers the cell?",
                    "back": "ATP",
                    "notes": None,
                    "extra": None,
                    "is_cloze": False,
                    "tags": ["biology"],
                    "ef": 2.5,
                    "interval_days": 0,
                    "repetitions": 0,
                    "lapses": 0,
                    "due_at": None,
                    "last_reviewed_at": None,
                    "queue_state": "new",
                    "created_at": "2026-04-20T00:00:00Z",
                    "last_modified": "2026-04-20T00:01:00Z",
                    "deleted": False,
                    "client_id": "server-client",
                    "version": 1,
                    "model_type": "basic",
                    "reverse": False,
                    "scheduler_type": "fsrs",
                    "next_intervals": {"again": "10m", "good": "1d"},
                },
                "selection_reason": "new",
            },
            {
                "uuid": "87ca2b3f-7e3a-47d7-a52f-8debc86c03cb",
                "ef": 2.6,
                "interval_days": 3,
                "repetitions": 1,
                "lapses": 0,
                "due_at": "2026-04-23T00:00:00Z",
                "last_reviewed_at": "2026-04-20T00:05:00Z",
                "last_modified": "2026-04-20T00:05:00Z",
                "version": 2,
                "scheduler_type": "fsrs",
                "queue_state": "review",
                "next_intervals": {"again": "10m", "good": "3d"},
                "review_session_id": 41,
            },
            {
                "id": 41,
                "deck_id": 7,
                "review_mode": "due",
                "scope_key": "due:deck:7",
                "status": "completed",
                "started_at": "2026-04-20T00:05:00Z",
                "completed_at": "2026-04-20T00:08:00Z",
                "client_id": "server-client",
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    created_deck = await client.create_flashcard_deck(
        FlashcardDeckCreateRequest(name="Biology", description="Cell review", scheduler_type="fsrs")
    )
    listed_decks = await client.list_flashcard_decks(limit=10, offset=5)
    created_card = await client.create_flashcard(
        FlashcardCreateRequest(
            deck_id=7,
            front="What powers the cell?",
            back="ATP",
            tags=["biology"],
        )
    )
    listed_cards = await client.list_flashcards(deck_id=7, q="cell", limit=10, offset=1)
    candidate = await client.get_next_flashcard_review(deck_id=7)
    reviewed = await client.review_flashcard(
        FlashcardReviewRequest(card_uuid="87ca2b3f-7e3a-47d7-a52f-8debc86c03cb", rating=4)
    )
    ended = await client.end_flashcard_review_session(41)

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/flashcards/decks")
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/flashcards/decks")
    assert mocked.await_args_list[1].kwargs["params"] == {"limit": 10, "offset": 5}
    assert mocked.await_args_list[2].args[:2] == ("POST", "/api/v1/flashcards")
    assert mocked.await_args_list[3].args[:2] == ("GET", "/api/v1/flashcards")
    assert mocked.await_args_list[3].kwargs["params"] == {
        "deck_id": 7,
        "q": "cell",
        "limit": 10,
        "offset": 1,
    }
    assert mocked.await_args_list[4].args[:2] == ("GET", "/api/v1/flashcards/review/next")
    assert mocked.await_args_list[4].kwargs["params"] == {"deck_id": 7}
    assert mocked.await_args_list[5].args[:2] == ("POST", "/api/v1/flashcards/review")
    assert mocked.await_args_list[6].args[:2] == ("POST", "/api/v1/flashcards/review-sessions/end")
    assert mocked.await_args_list[6].kwargs["json_data"] == {"review_session_id": 41}

    assert isinstance(created_deck, FlashcardDeckResponse)
    assert isinstance(created_card, FlashcardResponse)
    assert isinstance(listed_cards, FlashcardListResponse)
    assert isinstance(candidate, FlashcardNextReviewResponse)
    assert isinstance(reviewed, FlashcardReviewResponse)
    assert ended.id == 41
    assert listed_decks[0].name == "Biology"


@pytest.mark.asyncio
async def test_flashcard_update_and_delete_routes_wire_correctly(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
                "uuid": "87ca2b3f-7e3a-47d7-a52f-8debc86c03cb",
                "deck_id": 7,
                "front": "Updated question",
                "back": "Updated answer",
                "notes": None,
                "extra": None,
                "is_cloze": False,
                "tags": ["biology"],
                "ef": 2.5,
                "interval_days": 0,
                "repetitions": 0,
                "lapses": 0,
                "due_at": None,
                "last_reviewed_at": None,
                "queue_state": "new",
                "created_at": "2026-04-20T00:00:00Z",
                "last_modified": "2026-04-20T00:01:00Z",
                "deleted": False,
                "client_id": "server-client",
                "version": 3,
                "model_type": "basic",
                "reverse": False,
                "scheduler_type": "sm2_plus",
            },
            {"deleted": True},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    updated = await client.update_flashcard(
        "card-server-1",
        FlashcardUpdateRequest(front="Updated question", expected_version=3),
    )
    deleted = await client.delete_flashcard("card-server-1", expected_version=4)

    assert mocked.await_args_list[0].args[:2] == ("PATCH", "/api/v1/flashcards/card-server-1")
    assert mocked.await_args_list[0].kwargs["json_data"] == {
        "front": "Updated question",
        "expected_version": 3,
    }
    assert mocked.await_args_list[1].args[:2] == ("DELETE", "/api/v1/flashcards/card-server-1")
    assert mocked.await_args_list[1].kwargs["params"] == {"expected_version": 4}
    assert isinstance(updated, FlashcardResponse)
    assert deleted == {"deleted": True}


@pytest.mark.asyncio
async def test_flashcard_extended_routes_wire_and_return_typed_models(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            _deck_payload(name="Biology updated", version=2),
            {"items": [_card_payload()], "count": 1, "total": 1},
            {"results": [{"uuid": CARD_UUID, "status": "updated", "flashcard": _card_payload(version=2)}]},
            {"items": [{"tag": "biology", "count": 3}], "count": 1},
            {
                "reviewed_today": 4,
                "retention_rate_today": 0.75,
                "lapse_rate_today": 0.1,
                "avg_answer_time_ms_today": 1300.0,
                "study_streak_days": 3,
                "generated_at": "2026-04-20T00:00:00Z",
                "decks": [
                    {"deck_id": 7, "deck_name": "Biology", "total": 10, "new": 2, "learning": 1, "due": 3, "mature": 4}
                ],
            },
            _card_payload(),
            _card_payload(repetitions=0, version=2),
            _card_payload(tags=["biology", "cell"]),
            {"items": ["biology", "cell"], "count": 2},
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
            },
            {"imported": 1, "items": [{"uuid": CARD_UUID, "deck_id": 7}], "errors": []},
            [{"id": 41, "deck_id": 7, "review_mode": "due", "scope_key": "due:deck:7", "status": "completed"}],
            {
                "thread": _assistant_thread(),
                "messages": [_assistant_message()],
                "context_snapshot": {"card": {"uuid": CARD_UUID}},
                "available_actions": ["explain", "mnemonic"],
            },
            {
                "thread": _assistant_thread(message_count=2, version=2),
                "user_message": _assistant_message(id=13, role="user", content="Explain this"),
                "assistant_message": _assistant_message(id=14),
                "structured_payload": {},
                "context_snapshot": {"card": {"uuid": CARD_UUID}},
            },
            {
                "flashcards": [{"front": "What powers the cell?", "back": "ATP", "tags": ["biology"]}],
                "count": 1,
            },
            _template_payload(),
            {"items": [_template_payload()], "count": 1, "total": 1},
            _template_payload(),
            _template_payload(name="Updated template", version=2),
            {"deleted": True},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    deck = await client.update_flashcard_deck(
        7,
        FlashcardDeckUpdateRequest(name="Biology updated", expected_version=1),
    )
    created_bulk = await client.create_flashcards_bulk(
        [FlashcardCreateRequest(deck_id=7, front="What powers the cell?", back="ATP")]
    )
    updated_bulk = await client.update_flashcards_bulk(
        [FlashcardBulkUpdateItem(uuid=CARD_UUID, front="Updated", expected_version=1)]
    )
    tags = await client.list_flashcard_tag_suggestions(q="bio", limit=10)
    analytics = await client.get_flashcard_analytics_summary(deck_id=7, include_workspace_items=True)
    card = await client.get_flashcard(CARD_UUID)
    reset = await client.reset_flashcard_scheduling(CARD_UUID, FlashcardResetSchedulingRequest(expected_version=1))
    set_tags = await client.set_flashcard_tags(CARD_UUID, FlashcardTagsUpdate(tags=["biology", "cell"]))
    card_tags = await client.get_flashcard_tags(CARD_UUID)
    preview = await client.preview_structured_qa_import(
        StructuredQaImportPreviewRequest(content="Q: What powers the cell?\nA: ATP"),
        max_lines=10,
    )
    imported = await client.import_flashcards(FlashcardsImportRequest(content="Deck\tFront\tBack\nBio\tQ\tA", has_header=True))
    sessions = await client.list_flashcard_review_sessions(deck_id=7, status="completed")
    assistant = await client.get_flashcard_assistant(CARD_UUID)
    response = await client.respond_flashcard_assistant(
        CARD_UUID,
        StudyAssistantRespondRequest(action="explain", message="Explain this", expected_thread_version=1),
    )
    generated = await client.generate_flashcards(FlashcardGenerateRequest(text="Cells use ATP.", num_cards=1))
    template = await client.create_flashcard_template(
        FlashcardTemplateCreateRequest(
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
    )
    templates = await client.list_flashcard_templates(limit=10)
    fetched_template = await client.get_flashcard_template(3)
    updated_template = await client.update_flashcard_template(
        3,
        FlashcardTemplateUpdateRequest(name="Updated template", expected_version=1),
    )
    deleted_template = await client.delete_flashcard_template(3, expected_version=2)

    assert mocked.await_args_list[0].args[:2] == ("PATCH", "/api/v1/flashcards/decks/7")
    assert mocked.await_args_list[0].kwargs["json_data"] == {"name": "Biology updated", "expected_version": 1}
    assert mocked.await_args_list[1].args[:2] == ("POST", "/api/v1/flashcards/bulk")
    assert mocked.await_args_list[2].args[:2] == ("PATCH", "/api/v1/flashcards/bulk")
    assert mocked.await_args_list[3].args[:2] == ("GET", "/api/v1/flashcards/tags")
    assert mocked.await_args_list[3].kwargs["params"] == {"q": "bio", "limit": 10}
    assert mocked.await_args_list[4].args[:2] == ("GET", "/api/v1/flashcards/analytics/summary")
    assert mocked.await_args_list[4].kwargs["params"] == {
        "deck_id": 7,
        "include_workspace_items": True,
    }
    assert mocked.await_args_list[5].args[:2] == ("GET", f"/api/v1/flashcards/id/{CARD_UUID}")
    assert mocked.await_args_list[6].args[:2] == ("POST", f"/api/v1/flashcards/{CARD_UUID}/reset-scheduling")
    assert mocked.await_args_list[7].args[:2] == ("PUT", f"/api/v1/flashcards/{CARD_UUID}/tags")
    assert mocked.await_args_list[8].args[:2] == ("GET", f"/api/v1/flashcards/{CARD_UUID}/tags")
    assert mocked.await_args_list[9].args[:2] == ("POST", "/api/v1/flashcards/import/structured/preview")
    assert mocked.await_args_list[9].kwargs["params"] == {"max_lines": 10}
    assert mocked.await_args_list[10].args[:2] == ("POST", "/api/v1/flashcards/import")
    assert mocked.await_args_list[11].args[:2] == ("GET", "/api/v1/flashcards/review-sessions")
    assert mocked.await_args_list[11].kwargs["params"] == {"deck_id": 7, "status": "completed", "limit": 20}
    assert mocked.await_args_list[12].args[:2] == ("GET", f"/api/v1/flashcards/{CARD_UUID}/assistant")
    assert mocked.await_args_list[13].args[:2] == ("POST", f"/api/v1/flashcards/{CARD_UUID}/assistant/respond")
    assert mocked.await_args_list[14].args[:2] == ("POST", "/api/v1/flashcards/generate")
    assert mocked.await_args_list[15].args[:2] == ("POST", "/api/v1/flashcards/templates")
    assert mocked.await_args_list[16].args[:2] == ("GET", "/api/v1/flashcards/templates")
    assert mocked.await_args_list[16].kwargs["params"] == {"limit": 10, "offset": 0}
    assert mocked.await_args_list[17].args[:2] == ("GET", "/api/v1/flashcards/templates/3")
    assert mocked.await_args_list[18].args[:2] == ("PATCH", "/api/v1/flashcards/templates/3")
    assert mocked.await_args_list[19].args[:2] == ("DELETE", "/api/v1/flashcards/templates/3")
    assert mocked.await_args_list[19].kwargs["params"] == {"expected_version": 2}

    assert isinstance(deck, FlashcardDeckResponse)
    assert isinstance(created_bulk, FlashcardListResponse)
    assert isinstance(updated_bulk, FlashcardBulkUpdateResponse)
    assert isinstance(tags, FlashcardTagSuggestionsResponse)
    assert isinstance(analytics, FlashcardAnalyticsSummaryResponse)
    assert isinstance(card, FlashcardResponse)
    assert isinstance(reset, FlashcardResponse)
    assert isinstance(set_tags, FlashcardResponse)
    assert card_tags == {"items": ["biology", "cell"], "count": 2}
    assert isinstance(preview, StructuredQaImportPreviewResponse)
    assert imported["imported"] == 1
    assert isinstance(sessions[0], FlashcardReviewSessionSummary)
    assert isinstance(assistant, StudyAssistantContextResponse)
    assert isinstance(response, StudyAssistantRespondResponse)
    assert isinstance(generated, FlashcardGenerateResponse)
    assert isinstance(template, FlashcardTemplateResponse)
    assert isinstance(templates, FlashcardTemplateListResponse)
    assert isinstance(fetched_template, FlashcardTemplateResponse)
    assert updated_template.name == "Updated template"
    assert deleted_template == {"deleted": True}


@pytest.mark.asyncio
async def test_flashcard_file_routes_wire_multipart_and_binary(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    request_mock = AsyncMock(
        side_effect=[
            {
                "asset_uuid": CARD_UUID,
                "reference": f"flashcard-asset://{CARD_UUID}",
                "markdown_snippet": f"![image](flashcard-asset://{CARD_UUID})",
                "mime_type": "image/png",
                "byte_size": 7,
                "width": 1,
                "height": 1,
                "original_filename": "cell.png",
            },
            {"imported": 1, "items": [{"uuid": CARD_UUID, "deck_id": 7}], "errors": []},
            {"imported": 1, "items": [{"uuid": CARD_UUID, "deck_id": 7}], "errors": []},
        ]
    )
    binary_mock = AsyncMock(
        side_effect=[
            ReadingExportResponse(content=b"pngdata", content_type="image/png"),
            ReadingExportResponse(
                content=b"Deck\tFront\tBack\nBio\tQ\tA\n",
                content_type="text/tab-separated-values; charset=utf-8",
                content_disposition='attachment; filename="flashcards.tsv"',
                filename="flashcards.tsv",
            ),
        ]
    )
    monkeypatch.setattr(client, "_request", request_mock)
    monkeypatch.setattr(client, "_binary_request", binary_mock)

    uploaded = await client.upload_flashcard_asset(("cell.png", b"pngdata", "image/png"))
    content = await client.get_flashcard_asset_content(CARD_UUID)
    imported_json = await client.import_flashcards_json(("cards.json", b"[]", "application/json"), max_items=10)
    imported_apkg = await client.import_flashcards_apkg(("cards.apkg", b"apkg", "application/octet-stream"))
    exported = await client.export_flashcards(deck_id=7, format="csv", delimiter="\t", include_header=True)

    assert request_mock.await_args_list[0].args[:2] == ("POST", "/api/v1/flashcards/assets")
    assert request_mock.await_args_list[0].kwargs["files"] == [("file", ("cell.png", b"pngdata", "image/png"))]
    assert binary_mock.await_args_list[0].args[:2] == ("GET", f"/api/v1/flashcards/assets/{CARD_UUID}/content")
    assert request_mock.await_args_list[1].args[:2] == ("POST", "/api/v1/flashcards/import/json")
    assert request_mock.await_args_list[1].kwargs["params"] == {"max_items": 10}
    assert request_mock.await_args_list[2].args[:2] == ("POST", "/api/v1/flashcards/import/apkg")
    assert binary_mock.await_args_list[1].args[:2] == ("GET", "/api/v1/flashcards/export")
    assert binary_mock.await_args_list[1].kwargs["params"] == {
        "deck_id": 7,
        "format": "csv",
        "delimiter": "\t",
        "include_header": True,
    }

    assert isinstance(uploaded, FlashcardAssetMetadata)
    assert content.content == b"pngdata"
    assert imported_json["imported"] == 1
    assert imported_apkg["imported"] == 1
    assert exported.filename == "flashcards.tsv"


@pytest.mark.asyncio
async def test_study_pack_routes_wire_and_return_typed_models(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    study_pack_payload = {
        "id": 9,
        "workspace_id": "ws-1",
        "title": "Cell biology pack",
        "deck_id": 7,
        "source_bundle_json": {"source_items": [{"source_type": "note", "source_id": "note-1"}]},
        "generation_options_json": None,
        "status": "active",
        "superseded_by_pack_id": None,
        "created_at": "2026-04-21T00:00:00Z",
        "last_modified": "2026-04-21T00:01:00Z",
        "deleted": False,
        "client_id": "server-client",
        "version": 1,
    }
    job_payload = {
        "job": {
            "id": 42,
            "status": "queued",
            "domain": "study_pack",
            "queue": "study",
            "job_type": "generate_study_pack",
        }
    }
    mocked = AsyncMock(
        side_effect=[
            job_payload,
            {**job_payload, "job": {**job_payload["job"], "status": "completed"}, "study_pack": study_pack_payload},
            study_pack_payload,
            {**job_payload, "job": {**job_payload["job"], "id": 43}},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    request = StudyPackCreateJobRequest(
        title="Cell biology pack",
        workspace_id="ws-1",
        source_items=[StudyPackSourceSelection(source_type="note", source_id="note-1", label="Chapter notes")],
    )
    created = await client.create_study_pack_job(request)
    status = await client.get_study_pack_job_status(42)
    pack = await client.get_study_pack(9)
    regenerated = await client.regenerate_study_pack(9)

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/flashcards/study-packs/jobs")
    assert mocked.await_args_list[0].kwargs["json_data"] == {
        "title": "Cell biology pack",
        "workspace_id": "ws-1",
        "deck_mode": "new",
        "source_items": [
            {
                "source_type": "note",
                "source_id": "note-1",
                "label": "Chapter notes",
                "locator": {},
            }
        ],
    }
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/flashcards/study-packs/jobs/42")
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/flashcards/study-packs/9")
    assert mocked.await_args_list[3].args[:2] == ("POST", "/api/v1/flashcards/study-packs/9/regenerate")

    assert isinstance(created, StudyPackJobAcceptedResponse)
    assert isinstance(status, StudyPackJobStatusResponse)
    assert isinstance(pack, StudyPackSummaryResponse)
    assert isinstance(regenerated, StudyPackJobAcceptedResponse)
    assert status.study_pack is not None
    assert status.study_pack.id == 9
