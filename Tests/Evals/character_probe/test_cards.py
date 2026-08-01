import pytest

from tldw_chatbook.Evals.character_probe.cards import snapshot_cards
from tldw_chatbook.Evals.character_probe.models import CardSnapshot


class _FakeCharacterDB:
    """Stands in for CharactersRAGDB; only get_character_card_by_id is used."""

    def __init__(self, cards):
        self._cards = cards

    def get_character_card_by_id(self, character_id):
        return self._cards.get(character_id)


def _card(**overrides):
    base = {
        "id": 1,
        "name": "Vex",
        "system_prompt": "You are Vex.",
        "personality": "sardonic",
        "scenario": "a rooftop at night",
        "first_message": "You again.",
        "post_history_instructions": "Stay in character.",
        "message_example": "<START>",
    }
    base.update(overrides)
    return base


def test_snapshot_copies_every_field_used_in_prompting():
    db = _FakeCharacterDB({1: _card()})
    (snapshot,) = snapshot_cards(db, [1])
    assert snapshot == CardSnapshot(
        id=1,
        name="Vex",
        system_prompt="You are Vex.",
        personality="sardonic",
        scenario="a rooftop at night",
        first_message="You again.",
        post_history_instructions="Stay in character.",
        message_example="<START>",
    )


def test_missing_fields_become_empty_strings_not_none():
    db = _FakeCharacterDB({1: {"id": 1, "name": "Sparse"}})
    (snapshot,) = snapshot_cards(db, [1])
    assert snapshot.system_prompt == ""
    assert snapshot.first_message == ""


def test_order_follows_the_requested_ids():
    db = _FakeCharacterDB({1: _card(id=1, name="A"), 2: _card(id=2, name="B")})
    assert [c.name for c in snapshot_cards(db, [2, 1])] == ["B", "A"]


def test_a_missing_card_raises_naming_the_id():
    db = _FakeCharacterDB({1: _card()})
    with pytest.raises(ValueError, match="99"):
        snapshot_cards(db, [1, 99])


def test_no_ids_raises():
    with pytest.raises(ValueError, match="at least one character"):
        snapshot_cards(_FakeCharacterDB({}), [])
