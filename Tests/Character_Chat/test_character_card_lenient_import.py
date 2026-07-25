"""Regression tests for lenient character card import (task: char card import failures).

Covers:
- ``load_character_card_from_file`` actually parses JSON and PNG cards
  (previously a stub that always returned ``None``).
- V2 cards missing spec-``required`` fields (description, personality,
  scenario, first_mes, mes_example) still import.
- Un-namespaced extension keys and SillyTavern built-ins do not block import.
- Character books with numeric positions / missing entry fields are kept,
  with defaults applied instead of dropped.
- V3 (``ccv3``) PNG metadata is extracted.
- Cards without a usable ``name`` are still rejected.
"""

import base64
import json

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from tldw_chatbook.Character_Chat.Character_Chat_Lib import (
    extract_json_from_image_file,
    import_character_card_from_json_string,
    load_character_card_from_file,
    parse_character_book,
    validate_character_book,
    validate_v2_card,
)


def _v2_card(**data_overrides):
    data = {
        "name": "Test Char",
        "description": "A test character.",
        "personality": "Friendly.",
        "scenario": "A test scenario.",
        "first_mes": "Hello!",
        "mes_example": "",
    }
    data.update(data_overrides)
    return {"spec": "chara_card_v2", "spec_version": "2.0", "data": data}


def _write_png_with_metadata(path, metadata: dict):
    img = Image.new("RGB", (10, 10), color="red")
    png_info = PngInfo()
    for key, value in metadata.items():
        png_info.add_text(key, value)
    img.save(path, "PNG", pnginfo=png_info)
    return path


def _b64_json(payload: dict) -> str:
    return base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")


# ---------------------------------------------------------------------------
# load_character_card_from_file (was a None-returning stub)
# ---------------------------------------------------------------------------


def test_load_character_card_from_json_file(tmp_path):
    card_path = tmp_path / "card.json"
    card_path.write_text(json.dumps(_v2_card()), encoding="utf-8")

    parsed = load_character_card_from_file(card_path)

    assert parsed is not None
    assert parsed["name"] == "Test Char"
    assert parsed["first_message"] == "Hello!"


def test_load_character_card_from_png_file(tmp_path):
    png_path = _write_png_with_metadata(
        tmp_path / "card.png", {"chara": _b64_json(_v2_card())}
    )

    parsed = load_character_card_from_file(str(png_path))

    assert parsed is not None
    assert parsed["name"] == "Test Char"


def test_load_character_card_from_missing_file_returns_none(tmp_path):
    assert load_character_card_from_file(tmp_path / "nope.json") is None


def test_load_character_card_from_plain_image_returns_none(tmp_path):
    img_path = tmp_path / "plain.png"
    Image.new("RGB", (10, 10)).save(img_path, "PNG")
    assert load_character_card_from_file(img_path) is None


# ---------------------------------------------------------------------------
# Lenient V2 validation/parsing
# ---------------------------------------------------------------------------


def test_v2_card_missing_spec_required_fields_still_imports():
    card = _v2_card()
    for field in ("description", "personality", "scenario", "first_mes", "mes_example"):
        del card["data"][field]

    parsed = import_character_card_from_json_string(json.dumps(card))

    assert parsed is not None
    assert parsed["name"] == "Test Char"
    assert parsed["description"] == ""
    assert parsed["first_message"] == ""


def test_explicit_v2_card_with_validation_problems_no_longer_aborts():
    # Missing description/personality/etc. AND un-namespaced extension keys:
    # previously this hard-aborted with "Import aborted".
    card = _v2_card(
        description=None,
        personality=None,
        extensions={"world": "Lore", "talkativeness": "0.5", "fav": False},
    )
    parsed = import_character_card_from_json_string(json.dumps(card))
    assert parsed is not None
    assert parsed["name"] == "Test Char"
    assert parsed["extensions"]["world"] == "Lore"


def test_validate_v2_card_only_name_is_fatal():
    card = _v2_card()
    del card["data"]["description"]
    del card["data"]["first_mes"]
    is_valid, messages = validate_v2_card(card)
    assert is_valid is True
    assert messages  # warnings are still reported


def test_validate_v2_card_missing_name_is_fatal():
    card = _v2_card()
    del card["data"]["name"]
    is_valid, messages = validate_v2_card(card)
    assert is_valid is False
    assert any("name" in m for m in messages)


def test_v2_card_with_blank_name_still_rejected():
    blank = _v2_card()
    blank["data"]["name"] = "   "
    assert import_character_card_from_json_string(json.dumps(blank)) is None


def test_v3_spec_card_imports_leniently():
    card = _v2_card()
    card["spec"] = "chara_card_v3"
    card["spec_version"] = "3.0"
    parsed = import_character_card_from_json_string(json.dumps(card))
    assert parsed is not None
    assert parsed["name"] == "Test Char"


def test_non_string_fields_are_coerced():
    card = _v2_card(personality=["brave", "kind"], character_version=3)
    parsed = import_character_card_from_json_string(json.dumps(card))
    assert parsed is not None
    assert parsed["personality"] == "brave\nkind"
    assert parsed["character_version"] == "3"


# ---------------------------------------------------------------------------
# Character book leniency
# ---------------------------------------------------------------------------


def test_character_book_numeric_position_and_defaults_preserved():
    book = {
        "entries": [
            {"keys": ["dragon"], "content": "Dragons are real."},  # missing enabled/insertion_order
            {
                "keys": ["castle"],
                "content": "A castle nearby.",
                "enabled": True,
                "insertion_order": 5,
                "position": 1,  # SillyTavern numeric position
            },
        ]
    }
    is_valid, messages = validate_character_book(book)
    assert is_valid is True

    parsed = parse_character_book(book)
    assert len(parsed["entries"]) == 2
    first, second = parsed["entries"]
    assert first["enabled"] is True  # defaulted, not dropped
    assert first["insertion_order"] == 0  # defaults to list position
    assert second["position"] == "after_char"  # numeric 1 normalized
    assert second["insertion_order"] == 5


def test_character_book_entries_not_list_is_non_fatal():
    is_valid, messages = validate_character_book({"entries": "oops"})
    assert is_valid is True
    parsed = parse_character_book({"entries": "oops"})
    assert parsed["entries"] == []


# ---------------------------------------------------------------------------
# ccv3 (V3) PNG metadata extraction
# ---------------------------------------------------------------------------


def test_extract_ccv3_metadata_from_png(tmp_path):
    card = _v2_card()
    card["spec"] = "chara_card_v3"
    card["spec_version"] = "3.0"
    png_path = _write_png_with_metadata(
        tmp_path / "v3.png", {"ccv3": _b64_json(card)}
    )

    extracted = extract_json_from_image_file(str(png_path), str(tmp_path))

    assert extracted is not None
    assert json.loads(extracted)["data"]["name"] == "Test Char"


def test_load_v3_png_card_end_to_end(tmp_path):
    card = _v2_card()
    card["spec"] = "chara_card_v3"
    card["spec_version"] = "3.0"
    png_path = _write_png_with_metadata(
        tmp_path / "v3.png", {"ccv3": _b64_json(card)}
    )

    parsed = load_character_card_from_file(png_path)

    assert parsed is not None
    assert parsed["name"] == "Test Char"
