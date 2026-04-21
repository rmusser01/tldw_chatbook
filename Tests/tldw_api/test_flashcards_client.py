"""Tests for flashcard endpoint wiring on the shared TLDW API client."""

from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    FlashcardCreateRequest,
    FlashcardDeckCreateRequest,
    FlashcardDeckResponse,
    FlashcardListResponse,
    FlashcardNextReviewResponse,
    FlashcardResponse,
    FlashcardReviewRequest,
    FlashcardReviewResponse,
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
