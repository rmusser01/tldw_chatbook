import json

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, ConflictError


@pytest.fixture
def db_instance():
    db = CharactersRAGDB(":memory:", "test_study_client")
    try:
        yield db
    finally:
        db.close_connection()


def _create_card(db_instance):
    deck_id = db_instance.create_deck("Biology", "Cell review")
    card_id = db_instance.create_flashcard(
        {
            "deck_id": deck_id,
            "front": "What is ATP?",
            "back": "Energy currency",
            "tags": "biology cell",
            "type": "basic",
        }
    )
    return deck_id, card_id


def test_update_flashcard_updates_content_tags_metadata_and_version(db_instance):
    _, card_id = _create_card(db_instance)

    updated = db_instance.update_flashcard(
        card_id,
        front="What powers the cell?",
        back="ATP",
        tags=["biology", "energy"],
        notes="Review mitochondria.",
        extra="Cellular respiration",
        expected_version=1,
    )
    card = db_instance.get_flashcard(card_id)

    assert updated is True
    assert card is not None
    assert card["front"] == "What powers the cell?"
    assert card["back"] == "ATP"
    assert card["tags"] == "biology energy"
    assert json.loads(card["metadata"]) == {
        "notes": "Review mitochondria.",
        "extra": "Cellular respiration",
    }
    assert card["version"] == 2


def test_update_flashcard_rejects_stale_version_without_mutation(db_instance):
    _, card_id = _create_card(db_instance)

    with pytest.raises(ConflictError, match="Version mismatch updating flashcard"):
        db_instance.update_flashcard(card_id, front="Stale update", expected_version=0)

    card = db_instance.get_flashcard(card_id)

    assert card is not None
    assert card["front"] == "What is ATP?"
    assert card["version"] == 1
