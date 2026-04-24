"""Tests for flashcard endpoint wiring on the shared TLDW API client."""

from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    FlashcardAnalyticsSummaryResponse,
    FlashcardBulkUpdateItemRequest,
    FlashcardBulkUpdateResponse,
    FlashcardCreateRequest,
    FlashcardDeckCreateRequest,
    FlashcardDeckResponse,
    FlashcardDeckUpdateRequest,
    FlashcardListResponse,
    FlashcardNextReviewResponse,
    FlashcardResetSchedulingRequest,
    FlashcardResponse,
    FlashcardReviewRequest,
    FlashcardReviewResponse,
    FlashcardTagsResponse,
    FlashcardTagsUpdateRequest,
    FlashcardTagSuggestionsResponse,
    FlashcardTemplateCreateRequest,
    FlashcardTemplateListResponse,
    FlashcardTemplateResponse,
    FlashcardTemplateUpdateRequest,
    FlashcardUpdateRequest,
    TLDWAPIClient,
)


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
async def test_flashcard_management_routes_wire_and_return_typed_models(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    card_payload = {
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
        "version": 2,
        "model_type": "basic",
        "reverse": False,
        "scheduler_type": "fsrs",
    }
    mocked = AsyncMock(
        side_effect=[
            {
                "id": 7,
                "name": "Biology v2",
                "description": "Updated cell review",
                "workspace_id": None,
                "review_prompt_side": "back",
                "created_at": "2026-04-20T00:00:00Z",
                "last_modified": "2026-04-20T00:02:00Z",
                "deleted": False,
                "client_id": "server-client",
                "version": 2,
                "scheduler_type": "fsrs",
            },
            card_payload,
            {**card_payload, "version": 3, "queue_state": "new"},
            {**card_payload, "tags": ["biology", "cell"]},
            {"items": ["biology", "cell"], "count": 2},
            {
                "reviewed_today": 3,
                "retention_rate_today": 0.8,
                "lapse_rate_today": 0.1,
                "avg_answer_time_ms_today": 1200.0,
                "study_streak_days": 4,
                "generated_at": "2026-04-23T12:00:00Z",
                "decks": [
                    {
                        "deck_id": 7,
                        "deck_name": "Biology",
                        "total": 12,
                        "new": 4,
                        "learning": 2,
                        "due": 3,
                        "mature": 3,
                    }
                ],
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    deck = await client.update_flashcard_deck(
        7,
        FlashcardDeckUpdateRequest(
            name="Biology v2",
            description="Updated cell review",
            workspace_id=None,
            review_prompt_side="back",
            expected_version=1,
        ),
    )
    card = await client.get_flashcard("87ca2b3f-7e3a-47d7-a52f-8debc86c03cb")
    reset = await client.reset_flashcard_scheduling(
        "87ca2b3f-7e3a-47d7-a52f-8debc86c03cb",
        FlashcardResetSchedulingRequest(expected_version=2),
    )
    tagged = await client.set_flashcard_tags(
        "87ca2b3f-7e3a-47d7-a52f-8debc86c03cb",
        FlashcardTagsUpdateRequest(tags=["biology", "cell"]),
    )
    tags = await client.get_flashcard_tags("87ca2b3f-7e3a-47d7-a52f-8debc86c03cb")
    analytics = await client.get_flashcard_analytics_summary(
        deck_id=7,
        workspace_id="ws-1",
        include_workspace_items=True,
    )

    assert mocked.await_args_list[0].args[:2] == ("PATCH", "/api/v1/flashcards/decks/7")
    assert mocked.await_args_list[0].kwargs["json_data"] == {
        "name": "Biology v2",
        "description": "Updated cell review",
        "workspace_id": None,
        "review_prompt_side": "back",
        "expected_version": 1,
    }
    assert mocked.await_args_list[1].args[:2] == (
        "GET",
        "/api/v1/flashcards/id/87ca2b3f-7e3a-47d7-a52f-8debc86c03cb",
    )
    assert mocked.await_args_list[2].args[:2] == (
        "POST",
        "/api/v1/flashcards/87ca2b3f-7e3a-47d7-a52f-8debc86c03cb/reset-scheduling",
    )
    assert mocked.await_args_list[3].args[:2] == (
        "PUT",
        "/api/v1/flashcards/87ca2b3f-7e3a-47d7-a52f-8debc86c03cb/tags",
    )
    assert mocked.await_args_list[4].args[:2] == (
        "GET",
        "/api/v1/flashcards/87ca2b3f-7e3a-47d7-a52f-8debc86c03cb/tags",
    )
    assert mocked.await_args_list[5].args[:2] == (
        "GET",
        "/api/v1/flashcards/analytics/summary",
    )
    assert mocked.await_args_list[5].kwargs["params"] == {
        "deck_id": 7,
        "workspace_id": "ws-1",
        "include_workspace_items": True,
    }

    assert isinstance(deck, FlashcardDeckResponse)
    assert isinstance(card, FlashcardResponse)
    assert isinstance(reset, FlashcardResponse)
    assert isinstance(tagged, FlashcardResponse)
    assert isinstance(tags, FlashcardTagsResponse)
    assert isinstance(analytics, FlashcardAnalyticsSummaryResponse)
    assert deck.review_prompt_side == "back"
    assert tags.items == ["biology", "cell"]
    assert analytics.decks[0].deck_name == "Biology"


@pytest.mark.asyncio
async def test_flashcard_bulk_and_tag_suggestion_routes_wire_and_return_typed_models(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    card_payload = {
        "uuid": "87ca2b3f-7e3a-47d7-a52f-8debc86c03cb",
        "deck_id": 7,
        "front": "Question",
        "back": "Answer",
        "is_cloze": False,
        "tags": ["biology"],
        "ef": 2.5,
        "interval_days": 0,
        "repetitions": 0,
        "lapses": 0,
        "queue_state": "new",
        "deleted": False,
        "client_id": "server-client",
        "version": 2,
        "model_type": "basic",
        "reverse": False,
    }
    mocked = AsyncMock(
        side_effect=[
            {"items": [card_payload], "count": 1, "total": None},
            {
                "results": [
                    {
                        "uuid": "87ca2b3f-7e3a-47d7-a52f-8debc86c03cb",
                        "status": "updated",
                        "flashcard": {**card_payload, "tags": ["biology", "cell"], "version": 3},
                        "error": None,
                    }
                ]
            },
            {"items": [{"tag": "biology", "count": 3}], "count": 1},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    created = await client.create_flashcards_bulk(
        [
            FlashcardCreateRequest(
                deck_id=7,
                front="Question",
                back="Answer",
                tags=["biology"],
            )
        ]
    )
    updated = await client.update_flashcards_bulk(
        [
            FlashcardBulkUpdateItemRequest(
                uuid="87ca2b3f-7e3a-47d7-a52f-8debc86c03cb",
                tags=["biology", "cell"],
                expected_version=2,
            )
        ]
    )
    suggestions = await client.list_flashcard_tag_suggestions(q="bio", limit=10)

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/flashcards/bulk")
    assert mocked.await_args_list[0].kwargs["json_data"] == [
        {
            "deck_id": 7,
            "front": "Question",
            "back": "Answer",
            "is_cloze": False,
            "tags": ["biology"],
            "source_ref_type": "manual",
        }
    ]
    assert mocked.await_args_list[1].args[:2] == ("PATCH", "/api/v1/flashcards/bulk")
    assert mocked.await_args_list[1].kwargs["json_data"] == [
        {
            "tags": ["biology", "cell"],
            "expected_version": 2,
            "uuid": "87ca2b3f-7e3a-47d7-a52f-8debc86c03cb",
        }
    ]
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/flashcards/tags")
    assert mocked.await_args_list[2].kwargs["params"] == {"q": "bio", "limit": 10}
    assert isinstance(created, FlashcardListResponse)
    assert isinstance(updated, FlashcardBulkUpdateResponse)
    assert isinstance(suggestions, FlashcardTagSuggestionsResponse)
    assert updated.results[0].flashcard.tags == ["biology", "cell"]
    assert suggestions.items[0].count == 3


@pytest.mark.asyncio
async def test_flashcard_template_routes_wire_and_return_typed_models(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    template_payload = {
        "id": 12,
        "name": "Cloze Drill",
        "model_type": "cloze",
        "front_template": "{{statement}}",
        "back_template": None,
        "notes_template": "Focus: {{topic}}",
        "extra_template": None,
        "placeholder_definitions": [
            {
                "key": "statement",
                "label": "Statement",
                "required": True,
                "targets": ["front_template"],
            }
        ],
        "created_at": "2026-04-23T12:00:00Z",
        "last_modified": "2026-04-23T12:05:00Z",
        "deleted": False,
        "client_id": "server-client",
        "version": 2,
    }
    mocked = AsyncMock(
        side_effect=[
            template_payload,
            {"items": [template_payload], "count": 1, "total": 1},
            template_payload,
            {**template_payload, "notes_template": "Updated focus: {{topic}}", "version": 3},
            {"deleted": True},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    created = await client.create_flashcard_template(
        FlashcardTemplateCreateRequest(
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
    )
    listed = await client.list_flashcard_templates(limit=25, offset=5)
    fetched = await client.get_flashcard_template(12)
    updated = await client.update_flashcard_template(
        12,
        FlashcardTemplateUpdateRequest(notes_template="Updated focus: {{topic}}", expected_version=2),
    )
    deleted = await client.delete_flashcard_template(12, expected_version=3)

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/flashcards/templates")
    assert mocked.await_args_list[0].kwargs["json_data"] == {
        "name": "Cloze Drill",
        "model_type": "cloze",
        "front_template": "{{statement}}",
        "notes_template": "Focus: {{topic}}",
        "placeholder_definitions": [
            {
                "key": "statement",
                "label": "Statement",
                "required": True,
                "targets": ["front_template"],
            }
        ],
    }
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/flashcards/templates")
    assert mocked.await_args_list[1].kwargs["params"] == {"limit": 25, "offset": 5}
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/flashcards/templates/12")
    assert mocked.await_args_list[3].args[:2] == ("PATCH", "/api/v1/flashcards/templates/12")
    assert mocked.await_args_list[3].kwargs["json_data"] == {
        "notes_template": "Updated focus: {{topic}}",
        "expected_version": 2,
    }
    assert mocked.await_args_list[4].args[:2] == ("DELETE", "/api/v1/flashcards/templates/12")
    assert mocked.await_args_list[4].kwargs["params"] == {"expected_version": 3}

    assert isinstance(created, FlashcardTemplateResponse)
    assert isinstance(listed, FlashcardTemplateListResponse)
    assert isinstance(fetched, FlashcardTemplateResponse)
    assert isinstance(updated, FlashcardTemplateResponse)
    assert updated.notes_template == "Updated focus: {{topic}}"
    assert deleted == {"deleted": True}
