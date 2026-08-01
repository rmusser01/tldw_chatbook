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
        "description": "A dock-side fixer.",
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
        description="A dock-side fixer.",
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
    assert snapshot.description == ""


def test_description_is_snapshotted():
    """``description`` is a real ``character_cards`` column and the primary
    V2 persona field -- Console sends it. It was missing from
    _SNAPSHOT_FIELDS until the whole-branch review of task-1691 phase 1, so
    every probe ran against a character stripped of its main definition.
    Pinned here as well as in test_prompt.py because a snapshot that omits
    it is permanent: the run keeps no other copy of the card."""
    db = _FakeCharacterDB({1: _card(description="A dock-side fixer.")})
    (snapshot,) = snapshot_cards(db, [1])
    assert snapshot.description == "A dock-side fixer."


def test_every_snapshotted_field_is_composed_into_the_prompt():
    """Adding a field to _SNAPSHOT_FIELDS is only half a change: a field the
    snapshot carries but prompt.py never composes is text the model never
    sees, which is exactly how ``description`` went missing. Every text
    field's value must appear in the composed prompt (``first_message``
    seeds the assistant turn instead, so it is checked there)."""
    from tldw_chatbook.Evals.character_probe.prompt import build_messages

    values = {
        "description": "DESCRIPTION-TEXT",
        "system_prompt": "SYSTEM-TEXT",
        "personality": "PERSONALITY-TEXT",
        "scenario": "SCENARIO-TEXT",
        "post_history_instructions": "POST-TEXT",
        "message_example": "EXAMPLE-TEXT",
        "first_message": "GREETING-TEXT",
    }
    db = _FakeCharacterDB({1: _card(**values)})
    (snapshot,) = snapshot_cards(db, [1])
    messages = build_messages(snapshot, None, ["Hi?"], [])
    rendered = "\n".join(m["content"] for m in messages)
    for field, value in values.items():
        assert value in rendered, f"{field} never reaches the model"


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


class _MismatchedIdDB:
    """Returns a row whose own id disagrees with whatever id was requested."""

    def get_character_card_by_id(self, character_id):
        return _card(id=999999, name="Impostor")


def test_a_row_agreeing_with_the_requested_id_is_accepted():
    db = _FakeCharacterDB({1: _card(id=1, name="Vex")})
    (snapshot,) = snapshot_cards(db, [1])
    assert snapshot.id == 1
    assert snapshot.name == "Vex"


def test_a_row_reporting_a_different_id_raises_naming_both():
    with pytest.raises(ValueError, match="999999") as excinfo:
        snapshot_cards(_MismatchedIdDB(), [1])
    assert "1" in str(excinfo.value)
    assert "999999" in str(excinfo.value)


def test_string_id_is_rejected():
    db = _FakeCharacterDB({1: _card()})
    with pytest.raises(ValueError, match="int"):
        snapshot_cards(db, ["1"])


def test_bool_id_is_rejected():
    db = _FakeCharacterDB({1: _card()})
    with pytest.raises(ValueError, match="int"):
        snapshot_cards(db, [True])
