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

    def update_deck(
        self,
        deck_id,
        *,
        name=None,
        description=None,
        metadata=None,
        expected_version=None,
    ):
        self.calls.append(("update_deck", deck_id, name, description, metadata, expected_version))
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

    def update_flashcard(self, card_id, **kwargs):
        self.calls.append(("update_flashcard", card_id, kwargs))
        self.flashcards[card_id] = {**self.flashcards[card_id], **kwargs, "version": 3}
        return True

    def reset_flashcard_scheduling(self, card_id, *, expected_version=None):
        self.calls.append(("reset_flashcard_scheduling", card_id, expected_version))
        self.flashcards[card_id] = {
            **self.flashcards[card_id],
            "interval": 0,
            "repetitions": 0,
            "ease_factor": 2.5,
            "version": 3,
        }
        return True

    def set_flashcard_tags(self, card_id, *, tags):
        self.calls.append(("set_flashcard_tags", card_id, tags))
        self.flashcards[card_id] = {**self.flashcards[card_id], "tags": " ".join(tags), "version": 3}
        return True

    def get_flashcard_tags(self, card_id):
        self.calls.append(("get_flashcard_tags", card_id))
        return ["science", "biology"]

    def list_flashcard_tag_suggestions(self, *, q=None, limit=50):
        self.calls.append(("list_flashcard_tag_suggestions", q, limit))
        return [{"tag": "biology", "count": 3}]

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


def test_local_study_service_updates_deck_metadata_backed_settings():
    db = FakeDB()
    service = LocalStudyService(db=db)

    updated = service.update_deck(
        "deck-local-1",
        name="Biology v2",
        description="Updated",
        review_prompt_side="back",
        scheduler_type="sm2",
        scheduler_settings={"daily_limit": 20},
        expected_version=2,
    )

    assert updated["id"] == "deck-local-1"
    assert db.calls == [
        (
            "update_deck",
            "deck-local-1",
            "Biology v2",
            "Updated",
            {
                "review_prompt_side": "back",
                "scheduler_type": "sm2",
                "scheduler_settings": {"daily_limit": 20},
            },
            2,
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
                "id": "card-local-1",
                "front": "Question v2",
                "tags": ["science", "biology"],
                "expected_version": 2,
            }
        ]
    )

    assert created["count"] == 1
    assert created["items"][0]["id"] == "card-local-1"
    assert updated["count"] == 1
    assert updated["results"][0]["flashcard"]["version"] == 3


def test_local_study_service_resets_scheduling_and_manages_tags():
    db = FakeDB()
    service = LocalStudyService(db=db)

    reset = service.reset_flashcard_scheduling("card-local-1", expected_version=2)
    tagged = service.set_flashcard_tags("card-local-1", tags=["biology", "cell"])
    tags = service.get_flashcard_tags("card-local-1")
    suggestions = service.list_flashcard_tag_suggestions(q="bio", limit=10)

    assert reset["interval"] == 0
    assert tagged["tags"] == "biology cell"
    assert tags == {"items": ["science", "biology"], "count": 2}
    assert suggestions == {"items": [{"tag": "biology", "count": 3}], "count": 1}


def test_local_study_service_previews_imports_and_exports_flashcard_tsv():
    db = FakeDB()
    service = LocalStudyService(db=db)

    preview = service.preview_structured_qa_import("Q: What powers cells?\nA: ATP")
    imported = service.import_flashcards_tsv(
        "Deck\tFront\tBack\tTags\tNotes\nBiology\tWhat powers cells?\tATP\tbiology cell\tenergy",
        has_header=True,
    )
    exported = service.export_flashcards(deck_id="deck-local-1", include_header=True)

    assert preview == {
        "drafts": [
            {
                "front": "What powers cells?",
                "back": "ATP",
                "line_start": 1,
                "line_end": 2,
                "notes": None,
                "extra": None,
                "tags": [],
            }
        ],
        "errors": [],
        "detected_format": "qa_labels",
        "skipped_blocks": 0,
    }
    assert imported["imported"] == 1
    assert imported["items"] == [{"uuid": "card-local-1", "deck_id": "deck-local-1"}]
    assert exported.decode("utf-8").splitlines()[0] == "Deck\tFront\tBack\tTags\tNotes\tExtra"


def test_local_study_service_imports_flashcards_from_json_file(tmp_path):
    db = FakeDB()
    service = LocalStudyService(db=db)
    json_path = tmp_path / "cards.json"
    json_path.write_text(
        json.dumps(
            {
                "cards": [
                    {
                        "deck": "Biology",
                        "question": "What powers cells?",
                        "answer": "ATP",
                        "tags": ["biology", "cell"],
                        "notes": "energy",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    imported = service.import_flashcards_json_file(json_path, max_items=10, max_field_length=100)

    assert imported == {
        "imported": 1,
        "items": [{"uuid": "card-local-1", "deck_id": "deck-local-1"}],
        "errors": [],
    }
    assert ("create_flashcard", {
        "deck_id": "deck-local-1",
        "front": "What powers cells?",
        "back": "ATP",
        "tags": "biology cell",
        "type": "basic",
        "metadata": {"notes": "energy"},
    }) in db.calls


def test_local_study_service_persists_management_helpers_against_chachanotes_db(tmp_path):
    db = CharactersRAGDB(tmp_path / "study.db", client_id="test_client")
    service = LocalStudyService(db=db)
    deck_id = db.create_deck("Biology", "Cell review")
    card_id = db.create_flashcard(
        {
            "deck_id": deck_id,
            "front": "Question",
            "back": "Answer",
            "tags": "science",
            "type": "basic",
        }
    )

    deck = service.update_deck(
        deck_id,
        name="Biology v2",
        review_prompt_side="back",
        scheduler_type="sm2",
        expected_version=1,
    )
    reset = service.reset_flashcard_scheduling(card_id, expected_version=1)
    tagged = service.set_flashcard_tags(card_id, tags=["biology", "cell"])
    tags = service.get_flashcard_tags(card_id)
    suggestions = service.list_flashcard_tag_suggestions(q="bio", limit=5)
    bulk = service.update_flashcards_bulk(
        [
            {
                "id": card_id,
                "front": "Question v2",
                "expected_version": tagged["version"],
            }
        ]
    )

    assert deck["name"] == "Biology v2"
    assert json.loads(deck["metadata"])["scheduler_type"] == "sm2"
    assert reset["interval"] == 0
    assert tagged["tags"] == "biology cell"
    assert tags == {"items": ["biology", "cell"], "count": 2}
    assert suggestions["items"][0] == {"tag": "biology", "count": 1}
    assert bulk["results"][0]["flashcard"]["front"] == "Question v2"


def test_local_study_service_imports_and_exports_against_chachanotes_db(tmp_path):
    db = CharactersRAGDB(tmp_path / "study.db", client_id="test_client")
    service = LocalStudyService(db=db)

    preview = service.preview_structured_qa_import("Q: What powers cells?\nA: ATP")
    imported = service.import_flashcards_tsv(
        "Deck\tFront\tBack\tTags\tNotes\nBiology\tWhat powers cells?\tATP\tbio cell\tenergy",
        has_header=True,
    )
    exported = service.export_flashcards(deck_id=imported["items"][0]["deck_id"], include_header=True)
    exported_text = exported.decode("utf-8")

    assert preview["drafts"][0]["front"] == "What powers cells?"
    assert imported["imported"] == 1
    assert imported["errors"] == []
    assert "Deck\tFront\tBack\tTags\tNotes\tExtra" in exported_text
    assert "Biology\tWhat powers cells?\tATP\tbio cell\tenergy" in exported_text


def test_local_study_service_imports_json_against_chachanotes_db(tmp_path):
    db = CharactersRAGDB(tmp_path / "study.db", client_id="test_client")
    service = LocalStudyService(db=db)
    json_path = tmp_path / "cards.json"
    json_path.write_text(
        json.dumps(
            [
                {
                    "deck_name": "Biology",
                    "front": "What powers cells?",
                    "back": "ATP",
                    "tags": "bio,cell",
                    "extra": "mitochondria",
                }
            ]
        ),
        encoding="utf-8",
    )

    imported = service.import_flashcards_json_file(json_path)
    exported = service.export_flashcards(deck_id=imported["items"][0]["deck_id"], include_header=True)

    assert imported["imported"] == 1
    assert imported["errors"] == []
    assert "Biology\tWhat powers cells?\tATP\tbio cell\t\tmitochondria" in exported.decode("utf-8")


def test_local_study_service_dispatches_study_lifecycle_notifications(tmp_path):
    db = CharactersRAGDB(tmp_path / "study.db", client_id="test_client")
    notifications_db = ClientNotificationsDB(tmp_path / "notifications.db")
    dispatcher = NotificationDispatchService(notifications_db)
    service = LocalStudyService(
        db=db,
        notification_dispatch_service=dispatcher,
        notification_app=SimpleNamespace(),
    )

    deck = service.create_deck(name="Biology", description="Cell review")
    imported = service.import_flashcards_tsv(
        "Deck\tFront\tBack\nBiology\tWhat powers cells?\tATP",
        has_header=True,
    )
    exported = service.export_flashcards(deck_id=deck["id"], include_header=True)

    rows = notifications_db.list_notifications(limit=10, category="study")
    actions = {row["payload"]["action"] for row in rows}
    assert imported["imported"] == 1
    assert exported.startswith(b"Deck\tFront\tBack")
    assert {"deck_created", "flashcards_imported", "flashcards_exported"}.issubset(actions)
    assert all(row["source_backend"] == "local" for row in rows)


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
