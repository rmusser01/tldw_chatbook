"""TASK-38604: chub detection was unreachable (the v2 pattern matched every
``spec == "chara_card_v2"`` card first, so chub/rating/version_history were
silently dropped), and agnai persona objects were passed through as dicts,
later coerced to "".
"""

from __future__ import annotations

from tldw_chatbook.Character_Chat.character_card_formats import (
    detect_and_parse_character_card,
)


def _chub_card() -> dict:
    return {
        "spec": "chara_card_v2",
        "spec_version": "2.0",
        "data": {
            "name": "Chub Char",
            "description": "A chub-hosted character",
            "first_mes": "Hello from chub",
        },
        "chub": {"id": 42, "slug": "chub-char"},
        "rating": 4.5,
        "version_history": ["1.0", "1.1"],
    }


def test_chub_card_detected_as_chub_with_fields_preserved():
    parsed, fmt = detect_and_parse_character_card(_chub_card())

    assert fmt == "chub"
    assert parsed is not None
    ext = parsed["data"]["extensions"]
    assert ext["chub"] == {"id": 42, "slug": "chub-char"}
    assert ext["rating"] == 4.5
    assert ext["version_history"] == ["1.0", "1.1"]
    # The standard V2 body still parses
    assert parsed["data"]["name"] == "Chub Char"
    assert parsed["data"]["first_mes"] == "Hello from chub"


def test_plain_v2_card_still_detected_as_v2():
    card = {
        "spec": "chara_card_v2",
        "spec_version": "2.0",
        "data": {"name": "Plain", "description": "plain v2"},
    }
    parsed, fmt = detect_and_parse_character_card(card)

    assert fmt == "v2"
    assert parsed is not None and parsed["data"]["name"] == "Plain"


def test_agnai_persona_dict_survives_as_string():
    card = {
        "name": "Seraphina",
        "greeting": "Well met, traveler.",
        "persona": {"kind": "person", "attributes": {"species": "elf"}},
    }
    parsed, fmt = detect_and_parse_character_card(card)

    assert fmt == "agnai"
    assert parsed is not None
    personality = parsed["data"]["personality"]
    assert isinstance(personality, str)
    assert personality  # not coerced to ""
    assert "species" in personality and "elf" in personality


def test_agnai_persona_string_unchanged():
    card = {
        "name": "Ryk",
        "greeting": "Hail.",
        "persona": "gruff mercenary",
    }
    parsed, _fmt = detect_and_parse_character_card(card)

    assert parsed is not None
    assert parsed["data"]["personality"] == "gruff mercenary"
