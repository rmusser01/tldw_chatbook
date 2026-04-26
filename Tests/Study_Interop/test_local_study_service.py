import json
from types import SimpleNamespace

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notifications.client_notifications_db import ClientNotificationsDB
from tldw_chatbook.Notifications.notification_dispatch_service import NotificationDispatchService
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
                "tags": "science biology",
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

    def create_flashcard_template(
        self,
        *,
        name,
        model_type="basic",
        front_template,
        back_template=None,
        notes_template=None,
        extra_template=None,
        placeholder_definitions=None,
    ):
        self.calls.append(
            (
                "create_flashcard_template",
                name,
                model_type,
                front_template,
                back_template,
                notes_template,
                extra_template,
                placeholder_definitions,
            )
        )
        return {
            "id": "tmpl-local-1",
            "name": name,
            "model_type": model_type,
            "front_template": front_template,
            "back_template": back_template,
            "notes_template": notes_template,
            "extra_template": extra_template,
            "placeholder_definitions": placeholder_definitions or [],
            "version": 1,
        }

    def list_flashcard_templates(self, *, limit=100, offset=0):
        self.calls.append(("list_flashcard_templates", limit, offset))
        return {
            "items": [
                {
                    "id": "tmpl-local-1",
                    "name": "Cloze Drill",
                    "model_type": "cloze",
                    "front_template": "{{statement}}",
                    "placeholder_definitions": [],
                    "version": 1,
                }
            ],
            "count": 1,
        }

    def get_flashcard_template(self, template_id):
        self.calls.append(("get_flashcard_template", template_id))
        return {
            "id": template_id,
            "name": "Cloze Drill",
            "model_type": "cloze",
            "front_template": "{{statement}}",
            "placeholder_definitions": [],
            "version": 1,
        }

    def update_flashcard_template(
        self,
        template_id,
        *,
        name=None,
        model_type=None,
        front_template=None,
        back_template=None,
        notes_template=None,
        extra_template=None,
        placeholder_definitions=None,
        expected_version=None,
    ):
        self.calls.append(
            (
                "update_flashcard_template",
                template_id,
                name,
                model_type,
                front_template,
                back_template,
                notes_template,
                extra_template,
                placeholder_definitions,
                expected_version,
            )
        )
        return {
            "id": template_id,
            "name": name or "Cloze Drill",
            "model_type": model_type or "cloze",
            "front_template": front_template or "{{statement}}",
            "back_template": back_template,
            "notes_template": notes_template,
            "extra_template": extra_template,
            "placeholder_definitions": placeholder_definitions or [],
            "version": 2,
        }

    def delete_flashcard_template(self, template_id, *, expected_version):
        self.calls.append(("delete_flashcard_template", template_id, expected_version))
        return {"deleted": True}

    def create_flashcard_asset(self, *, original_filename, mime_type, content):
        self.calls.append(("create_flashcard_asset", original_filename, mime_type, content))
        return {
            "asset_uuid": "asset-local-1",
            "reference": "flashcard-asset://asset-local-1",
            "markdown_snippet": "![cell](flashcard-asset://asset-local-1)",
            "mime_type": mime_type,
            "byte_size": len(content),
            "width": None,
            "height": None,
            "original_filename": original_filename,
        }

    def get_flashcard_asset_content(self, asset_uuid):
        self.calls.append(("get_flashcard_asset_content", asset_uuid))
        return b"fake-png"


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


def test_local_study_service_fetches_flashcard_by_id():
    db = FakeDB()
    service = LocalStudyService(db=db)

    card = service.get_flashcard("card-local-1")

    assert card["id"] == "card-local-1"
    assert db.calls == [("get_flashcard", "card-local-1")]


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


def test_local_study_service_persists_flashcard_templates_against_chachanotes_db(tmp_path):
    db_path = tmp_path / "study.db"
    db = CharactersRAGDB(db_path, client_id="test_client")
    service = LocalStudyService(db=db)

    created = service.create_flashcard_template(
        name="Cloze Drill",
        model_type="cloze",
        front_template="{{statement}}",
        notes_template="Focus: {{topic}}",
        placeholder_definitions=[{"name": "statement", "targets": ["front_template"]}],
    )
    listed = service.list_flashcard_templates(limit=10, offset=0)
    updated = service.update_flashcard_template(
        created["id"],
        notes_template="Updated focus: {{topic}}",
        expected_version=created["version"],
    )
    fetched = service.get_flashcard_template(created["id"])
    db.close()

    reopened = CharactersRAGDB(db_path, client_id="test_client")
    reopened_service = LocalStudyService(db=reopened)
    reopened_fetched = reopened_service.get_flashcard_template(created["id"])
    deleted = reopened_service.delete_flashcard_template(
        created["id"],
        expected_version=reopened_fetched["version"],
    )
    after_delete = reopened_service.list_flashcard_templates(limit=10, offset=0)

    assert created["id"]
    assert created["version"] == 1
    assert created["placeholder_definitions"] == [{"name": "statement", "targets": ["front_template"]}]
    assert listed["items"][0]["id"] == created["id"]
    assert updated["notes_template"] == "Updated focus: {{topic}}"
    assert updated["version"] == created["version"] + 1
    assert fetched["notes_template"] == "Updated focus: {{topic}}"
    assert reopened_fetched["notes_template"] == "Updated focus: {{topic}}"
    assert deleted is True
    assert after_delete == {"items": [], "count": 0, "total": 0}


def test_local_study_service_persists_flashcard_assets_against_chachanotes_db(tmp_path):
    db_path = tmp_path / "study.db"
    image_path = tmp_path / "cell.png"
    image_path.write_bytes(b"fake-png")
    db = CharactersRAGDB(db_path, client_id="test_client")
    service = LocalStudyService(db=db)

    asset = service.upload_flashcard_asset(image_path)
    content = service.get_flashcard_asset_content(asset["asset_uuid"])
    db.close()

    reopened = CharactersRAGDB(db_path, client_id="test_client")
    reopened_service = LocalStudyService(db=reopened)
    reopened_content = reopened_service.get_flashcard_asset_content(asset["asset_uuid"])

    assert asset["asset_uuid"]
    assert asset["reference"] == f"flashcard-asset://{asset['asset_uuid']}"
    assert asset["markdown_snippet"] == f"![cell.png](flashcard-asset://{asset['asset_uuid']})"
    assert asset["mime_type"] == "image/png"
    assert asset["byte_size"] == len(b"fake-png")
    assert asset["original_filename"] == "cell.png"
    assert content == b"fake-png"
    assert reopened_content == b"fake-png"
