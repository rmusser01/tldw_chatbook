from tldw_chatbook.Study_Interop.local_study_service import LocalStudyService


class FakeDB:
    def __init__(self):
        self.calls = []
        self.flashcards = {
            "card-local-1": {
                "id": "card-local-1",
                "deck_id": "deck-local-1",
                "front": "Question",
                "back": "Answer",
                "version": 2,
            }
        }

    def list_decks(self, *, limit=100, offset=0):
        self.calls.append(("list_decks", limit, offset))
        return [{"id": "deck-local-1", "name": "Biology", "description": "Cell review"}]

    def create_deck(self, name, description=None):
        self.calls.append(("create_deck", name, description))
        return "deck-local-1"

    def get_deck(self, deck_id):
        self.calls.append(("get_deck", deck_id))
        return {"id": deck_id, "name": "Biology", "description": "Cell review"}

    def update_deck(self, deck_id, **payload):
        self.calls.append(("update_deck", deck_id, payload))
        return True

    def list_flashcards(self, *, deck_id=None, q=None, limit=100, offset=0):
        self.calls.append(("list_flashcards", deck_id, q, limit, offset))
        return [{"id": "card-local-1", "deck_id": deck_id, "front": "Question", "back": "Answer"}]

    def create_flashcard(self, payload):
        self.calls.append(("create_flashcard", payload))
        return "card-local-1"

    def get_flashcard(self, card_id):
        self.calls.append(("get_flashcard", card_id))
        return dict(self.flashcards[card_id])

    def update_flashcard(self, card_id, **payload):
        self.calls.append(("update_flashcard", card_id, payload))
        stored_payload = {
            key: (" ".join(value) if key == "tags" and isinstance(value, list) else value)
            for key, value in payload.items()
            if key != "expected_version" and value is not None
        }
        self.flashcards[card_id] = {
            **self.flashcards[card_id],
            **stored_payload,
            "version": self.flashcards[card_id]["version"] + 1,
        }
        return True

    def get_due_flashcards(self, *, deck_id=None, limit=20):
        self.calls.append(("get_due_flashcards", deck_id, limit))
        return [{"id": "card-local-1", "deck_id": deck_id, "front": "Question", "back": "Answer"}]

    def update_flashcard_review(self, card_id, rating):
        self.calls.append(("update_flashcard_review", card_id, rating))

    def delete_flashcard(self, card_id, *, expected_version=None, hard_delete=False):
        self.calls.append(("delete_flashcard", card_id, expected_version, hard_delete))
        return True

    def move_flashcard(self, card_id, target_deck_id, *, expected_version=None):
        self.calls.append(("move_flashcard", card_id, target_deck_id, expected_version))
        self.flashcards[card_id] = {
            **self.flashcards[card_id],
            "deck_id": target_deck_id,
            "version": self.flashcards[card_id]["version"] + 1,
        }
        return True

    def delete_deck(self, deck_id, *, expected_version=None, hard_delete=False):
        self.calls.append(("delete_deck", deck_id, expected_version, hard_delete))
        return True


def test_local_study_service_lists_and_creates_decks():
    db = FakeDB()
    service = LocalStudyService(db=db)

    listed = service.list_decks(limit=5, offset=1)
    created = service.create_deck(name="Biology", description="Cell review")

    assert listed[0]["name"] == "Biology"
    assert created["id"] == "deck-local-1"
    assert db.calls == [
        ("list_decks", 5, 1),
        ("create_deck", "Biology", "Cell review"),
        ("get_deck", "deck-local-1"),
    ]


def test_local_study_service_updates_decks_and_refetches_record():
    db = FakeDB()
    service = LocalStudyService(db=db)

    updated = service.update_deck(
        "deck-local-1",
        name="Biology Updated",
        description="Cells and genetics",
        expected_version=3,
    )

    assert updated["id"] == "deck-local-1"
    assert db.calls == [
        (
            "update_deck",
            "deck-local-1",
            {"name": "Biology Updated", "description": "Cells and genetics", "expected_version": 3},
        ),
        ("get_deck", "deck-local-1"),
    ]


def test_local_study_service_normalizes_blank_search_to_list_query():
    db = FakeDB()
    service = LocalStudyService(db=db)

    listed = service.list_flashcards(deck_id="deck-local-1", q="   ", limit=7, offset=3)

    assert listed[0]["deck_id"] == "deck-local-1"
    assert db.calls == [("list_flashcards", "deck-local-1", None, 7, 3)]


def test_local_study_service_creates_flashcards_and_fetches_due_review_card():
    db = FakeDB()
    service = LocalStudyService(db=db)

    created = service.create_flashcard(
        deck_id="deck-local-1",
        front="Question",
        back="Answer",
        tags=["science", "biology"],
    )
    candidate = service.get_next_review_candidate(deck_id="deck-local-1")

    assert created["id"] == "card-local-1"
    assert candidate["card"]["id"] == "card-local-1"
    assert candidate["selection_reason"] == "due"


def test_local_study_service_gets_and_sets_flashcard_tags():
    db = FakeDB()
    db.flashcards["card-local-1"]["tags"] = "science biology science"
    service = LocalStudyService(db=db)

    tags = service.get_flashcard_tags("card-local-1")
    tagged = service.set_flashcard_tags("card-local-1", tags=["biology", "cells"])

    assert tags == {"uuid": "card-local-1", "tags": ["science", "biology"]}
    assert tagged["tags"] == "biology cells"
    assert db.calls == [
        ("get_flashcard", "card-local-1"),
        ("update_flashcard", "card-local-1", {"tags": ["biology", "cells"]}),
        ("get_flashcard", "card-local-1"),
    ]


def test_local_study_service_bulk_creates_and_updates_flashcards():
    db = FakeDB()
    service = LocalStudyService(db=db)

    created = service.create_flashcards_bulk(
        [
            {
                "deck_id": "deck-local-1",
                "front": "Question",
                "back": "Answer",
                "tags": ["science"],
            }
        ]
    )
    updated = service.update_flashcards_bulk(
        [
            {
                "uuid": "card-local-1",
                "front": "Updated question",
                "tags": ["science", "updated"],
                "expected_version": 2,
            }
        ]
    )

    assert created["count"] == 1
    assert created["items"][0]["id"] == "card-local-1"
    assert updated["results"] == [
        {
            "uuid": "card-local-1",
            "status": "updated",
            "flashcard": {
                "id": "card-local-1",
                "deck_id": "deck-local-1",
                "front": "Updated question",
                "back": "Answer",
                "tags": "science updated",
                "version": 3,
            },
        }
    ]


def test_local_study_service_updates_review_and_returns_refetched_card():
    db = FakeDB()
    service = LocalStudyService(db=db)

    outcome = service.submit_flashcard_review("card-local-1", rating=4)

    assert outcome["card"]["id"] == "card-local-1"
    assert outcome["rating"] == 4
    assert db.calls == [
        ("update_flashcard_review", "card-local-1", 4),
        ("get_flashcard", "card-local-1"),
    ]


def test_local_study_service_deletes_and_moves_cards_with_expected_version():
    db = FakeDB()
    service = LocalStudyService(db=db)

    deleted = service.delete_flashcard("card-local-1", expected_version=2)
    moved = service.move_flashcard("card-local-1", target_deck_id="deck-local-2", expected_version=2)

    assert deleted is True
    assert moved["deck_id"] == "deck-local-2"
    assert moved["version"] == 3
    assert db.calls == [
        ("delete_flashcard", "card-local-1", 2, False),
        ("move_flashcard", "card-local-1", "deck-local-2", 2),
        ("get_flashcard", "card-local-1"),
    ]


def test_local_study_service_deletes_deck_with_expected_version():
    db = FakeDB()
    service = LocalStudyService(db=db)

    deleted = service.delete_deck("deck-local-1", expected_version=4)

    assert deleted is True
    assert db.calls == [("delete_deck", "deck-local-1", 4, False)]
