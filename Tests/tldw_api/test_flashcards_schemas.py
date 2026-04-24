import pytest
from pydantic import ValidationError

from tldw_chatbook.tldw_api import (
    FlashcardBulkUpdateItemRequest,
    FlashcardBulkUpdateResponse,
    FlashcardCreateRequest,
    FlashcardDeckCreateRequest,
    FlashcardDeckResponse,
    FlashcardListResponse,
    FlashcardNextReviewResponse,
    FlashcardResponse,
    FlashcardReviewRequest,
    FlashcardReviewResponse,
    FlashcardTagSuggestionsResponse,
    FlashcardTemplateCreateRequest,
    FlashcardTemplateListResponse,
    FlashcardTemplateResponse,
    FlashcardTemplateUpdateRequest,
    FlashcardUpdateRequest,
)


def test_flashcard_deck_create_request_round_trips_workspace_and_scheduler_fields():
    payload = FlashcardDeckCreateRequest(
        name="Biology",
        description="Cell review",
        workspace_id="ws-1",
        scheduler_type="fsrs",
    )

    dumped = payload.model_dump(exclude_none=True)
    assert dumped == {
        "name": "Biology",
        "description": "Cell review",
        "workspace_id": "ws-1",
        "scheduler_type": "fsrs",
    }


def test_flashcard_deck_response_preserves_workspace_and_metadata_defaults():
    payload = FlashcardDeckResponse.model_validate(
        {
            "id": 7,
            "name": "Biology",
            "description": "Cell review",
            "workspace_id": "ws-1",
            "created_at": "2026-04-20T00:00:00Z",
            "last_modified": "2026-04-20T00:05:00Z",
            "deleted": False,
            "client_id": "server-client",
            "version": 2,
            "scheduler_type": "fsrs",
        }
    )

    assert payload.id == 7
    assert payload.workspace_id == "ws-1"
    assert payload.scheduler_type == "fsrs"


def test_flashcard_response_populates_tags_from_tags_json():
    payload = FlashcardResponse.model_validate(
        {
            "uuid": "87ca2b3f-7e3a-47d7-a52f-8debc86c03cb",
            "deck_id": 7,
            "front": "What powers the cell?",
            "back": "ATP",
            "notes": None,
            "extra": None,
            "is_cloze": False,
            "tags_json": '["biology", "cells"]',
            "ef": 2.5,
            "interval_days": 3,
            "repetitions": 2,
            "lapses": 0,
            "due_at": "2026-04-21T00:00:00Z",
            "last_reviewed_at": "2026-04-20T00:00:00Z",
            "queue_state": "review",
            "created_at": "2026-04-20T00:00:00Z",
            "last_modified": "2026-04-20T00:05:00Z",
            "deleted": False,
            "client_id": "server-client",
            "version": 4,
            "model_type": "basic",
            "reverse": False,
            "scheduler_type": "fsrs",
        }
    )

    assert payload.tags == ["biology", "cells"]
    assert payload.queue_state == "review"
    assert payload.scheduler_type == "fsrs"


def test_flashcard_list_response_wraps_flashcard_records():
    payload = FlashcardListResponse.model_validate(
        {
            "items": [
                {
                    "uuid": "87ca2b3f-7e3a-47d7-a52f-8debc86c03cb",
                    "deck_id": 7,
                    "front": "Question",
                    "back": "Answer",
                    "notes": None,
                    "extra": None,
                    "is_cloze": False,
                    "tags": ["science"],
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
                    "scheduler_type": "sm2_plus",
                }
            ],
            "count": 1,
            "total": 1,
        }
    )

    assert payload.count == 1
    assert payload.items[0].front == "Question"


def test_flashcard_review_request_and_response_preserve_session_and_intervals():
    request_data = FlashcardReviewRequest(card_uuid="87ca2b3f-7e3a-47d7-a52f-8debc86c03cb", rating=4)
    assert request_data.model_dump() == {
        "card_uuid": "87ca2b3f-7e3a-47d7-a52f-8debc86c03cb",
        "rating": 4,
        "answer_time_ms": None,
    }

    payload = FlashcardReviewResponse.model_validate(
        {
            "uuid": "87ca2b3f-7e3a-47d7-a52f-8debc86c03cb",
            "ef": 2.7,
            "interval_days": 5,
            "repetitions": 3,
            "lapses": 0,
            "due_at": "2026-04-25T00:00:00Z",
            "last_reviewed_at": "2026-04-20T00:05:00Z",
            "last_modified": "2026-04-20T00:05:00Z",
            "version": 5,
            "scheduler_type": "fsrs",
            "queue_state": "review",
            "next_intervals": {"again": "10m", "good": "5d"},
            "review_session_id": 41,
        }
    )

    assert payload.next_intervals["good"] == "5d"
    assert payload.review_session_id == 41


def test_flashcard_next_review_response_defaults_to_no_card():
    payload = FlashcardNextReviewResponse.model_validate({"card": None, "selection_reason": "none"})
    assert payload.card is None
    assert payload.selection_reason == "none"


def test_flashcard_template_models_round_trip_server_shape():
    create_payload = FlashcardTemplateCreateRequest(
        name="Cloze Drill",
        model_type="cloze",
        front_template="{{statement}}",
        notes_template="Focus: {{topic}}",
        placeholder_definitions=[
            {
                "key": "statement",
                "label": "Statement",
                "required": True,
                "targets": ["front_template"],
            }
        ],
    )
    update_payload = FlashcardTemplateUpdateRequest(
        notes_template="Updated focus: {{topic}}",
        expected_version=2,
    )
    response = FlashcardTemplateResponse.model_validate(
        {
            "id": 12,
            "name": "Cloze Drill",
            "model_type": "cloze",
            "front_template": "{{statement}}",
            "back_template": None,
            "notes_template": "Focus: {{topic}}",
            "extra_template": None,
            "placeholder_definitions": create_payload.placeholder_definitions,
            "created_at": "2026-04-23T12:00:00Z",
            "last_modified": "2026-04-23T12:05:00Z",
            "deleted": False,
            "client_id": "server-client",
            "version": 2,
        }
    )
    listed = FlashcardTemplateListResponse.model_validate({"items": [response], "count": 1, "total": 1})

    assert create_payload.model_dump(mode="json", exclude_none=True)["placeholder_definitions"][0]["key"] == "statement"
    assert update_payload.model_dump(mode="json", exclude_none=True) == {
        "notes_template": "Updated focus: {{topic}}",
        "expected_version": 2,
    }
    assert response.placeholder_definitions[0].targets == ["front_template"]
    assert listed.items[0].id == 12


def test_flashcard_bulk_update_and_tag_suggestion_models_round_trip_server_shape():
    update_item = FlashcardBulkUpdateItemRequest(
        uuid="87ca2b3f-7e3a-47d7-a52f-8debc86c03cb",
        tags=["biology", "cell"],
        expected_version=2,
    )
    bulk_response = FlashcardBulkUpdateResponse.model_validate(
        {
            "results": [
                {
                    "uuid": "87ca2b3f-7e3a-47d7-a52f-8debc86c03cb",
                    "status": "validation_error",
                    "flashcard": None,
                    "error": {
                        "code": "validation_error",
                        "message": "Invalid deck",
                        "invalid_deck_ids": [404],
                    },
                }
            ]
        }
    )
    suggestions = FlashcardTagSuggestionsResponse.model_validate(
        {"items": [{"tag": "biology", "count": 3}], "count": 1}
    )

    assert update_item.model_dump(mode="json", exclude_none=True) == {
        "tags": ["biology", "cell"],
        "expected_version": 2,
        "uuid": "87ca2b3f-7e3a-47d7-a52f-8debc86c03cb",
    }
    assert bulk_response.results[0].status == "validation_error"
    assert bulk_response.results[0].error.invalid_deck_ids == [404]
    assert suggestions.items[0].tag == "biology"


def test_flashcard_create_request_preserves_optional_fields():
    payload = FlashcardCreateRequest(
        deck_id=7,
        front="What powers the cell?",
        back="ATP",
        notes="Chapter 1",
        extra="Mitochondria",
        tags=["biology"],
        model_type="basic",
    )

    assert payload.model_dump(exclude_none=True) == {
        "deck_id": 7,
        "front": "What powers the cell?",
        "back": "ATP",
        "notes": "Chapter 1",
        "extra": "Mitochondria",
        "tags": ["biology"],
        "model_type": "basic",
        "is_cloze": False,
        "source_ref_type": "manual",
    }


def test_flashcard_update_request_accepts_expected_version():
    payload = FlashcardUpdateRequest(deck_id=7, expected_version=3)

    assert payload.model_dump(exclude_none=True)["expected_version"] == 3


def test_flashcard_update_request_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        FlashcardUpdateRequest(deck_id=7, unexpected_field="nope")


def test_flashcard_update_request_rejects_non_positive_expected_version():
    with pytest.raises(ValidationError):
        FlashcardUpdateRequest(deck_id=7, expected_version=0)


def test_flashcard_list_response_accepts_sm2_plus_scheduler_type():
    payload = FlashcardListResponse.model_validate(
        {
            "items": [
                {
                    "uuid": "87ca2b3f-7e3a-47d7-a52f-8debc86c03cb",
                    "deck_id": 7,
                    "front": "Question",
                    "back": "Answer",
                    "notes": None,
                    "extra": None,
                    "is_cloze": False,
                    "tags": ["science"],
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
                    "scheduler_type": "sm2_plus",
                }
            ],
            "count": 1,
            "total": 1,
        }
    )

    assert payload.items[0].scheduler_type == "sm2_plus"
