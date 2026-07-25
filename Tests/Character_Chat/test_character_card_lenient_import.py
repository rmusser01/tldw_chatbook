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


# ---------------------------------------------------------------------------
# Review follow-ups: nameless cards, mutation safety, WebP EXIF
# ---------------------------------------------------------------------------


def test_nameless_v2_card_is_rejected_not_named_unknown():
    # A V2 card with no name anywhere must NOT import as placeholder "Unknown".
    card = {"spec": "chara_card_v2", "spec_version": "2.0", "data": {
        "description": "No name here.",
        "first_mes": "Hello?",
    }}
    assert import_character_card_from_json_string(json.dumps(card)) is None


def test_nameless_flat_card_is_rejected_not_named_unknown():
    # Flat card with no name-like field at all must be rejected too.
    card = {"description": "Just a description.", "first_mes": "Hi."}
    assert import_character_card_from_json_string(json.dumps(card)) is None


def test_generic_fallback_rescues_name_from_alternate_fields():
    # CharacterAI-style export: name lives in participant__name, not 'name'.
    card = {
        "participant__name": "CAI Hero",
        "greeting": "Greetings, traveler.",
        "description": "A CharacterAI export.",
    }
    parsed = import_character_card_from_json_string(json.dumps(card))
    assert parsed is not None
    assert parsed["name"] == "CAI Hero"


def test_two_nameless_cards_do_not_merge_into_one_character(tmp_path):
    # Regression: previously two distinct nameless cards both parsed as
    # "Unknown"; the second then hit name-conflict resolution and silently
    # resolved to the first character's ID.
    from tldw_chatbook.Character_Chat.Character_Chat_Lib import (
        import_and_save_character_from_file,
    )
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(tmp_path / "nameless.db", "test-client")
    card_a = tmp_path / "a.json"
    card_b = tmp_path / "b.json"
    card_a.write_text(json.dumps({"spec": "chara_card_v2", "spec_version": "2.0",
                                  "data": {"description": "Card A"}}), encoding="utf-8")
    card_b.write_text(json.dumps({"spec": "chara_card_v2", "spec_version": "2.0",
                                  "data": {"description": "Card B"}}), encoding="utf-8")

    id_a = import_and_save_character_from_file(db, str(card_a))
    id_b = import_and_save_character_from_file(db, str(card_b))

    assert id_a is None
    assert id_b is None
    # The DB seeds a default character on init; the regression check is that
    # no placeholder "Unknown" character was created from the nameless cards.
    names = [c["name"] for c in db.list_character_cards()]
    assert "Unknown" not in names


def test_parse_v2_card_does_not_mutate_input_extensions():
    card = _v2_card(extensions={"chub": {"id": 1}})
    card["data"]["character_book"] = {"entries": []}
    original_extensions = card["data"]["extensions"]

    parsed = import_character_card_from_json_string(json.dumps(card))

    assert parsed is not None
    # character_book must land in the parsed extensions...
    assert "character_book" in parsed["extensions"]
    # ...but the caller's original dict must be untouched.
    assert "character_book" not in original_extensions


def test_dict_valued_text_field_defaults_to_empty_string():
    card = _v2_card(description={"unexpected": "mapping"})
    parsed = import_character_card_from_json_string(json.dumps(card))
    assert parsed is not None
    assert parsed["description"] == ""


def _write_webp_with_exif_chara(path, payload: dict):
    img = Image.new("RGB", (10, 10), color="blue")
    exif = Image.Exif()
    # SillyTavern-style: UserComment carries an ASCII charset prefix + base64.
    exif[37510] = b"ASCII\x00\x00\x00" + _b64_json(payload).encode("ascii")
    img.save(path, "WEBP", exif=exif)
    return path


def test_extract_chara_from_webp_exif_user_comment(tmp_path):
    webp_path = _write_webp_with_exif_chara(tmp_path / "card.webp", _v2_card())

    extracted = extract_json_from_image_file(str(webp_path), str(tmp_path))

    assert extracted is not None
    assert json.loads(extracted)["data"]["name"] == "Test Char"


def test_load_webp_card_end_to_end(tmp_path):
    webp_path = _write_webp_with_exif_chara(tmp_path / "card.webp", _v2_card())

    parsed = load_character_card_from_file(webp_path)

    assert parsed is not None
    assert parsed["name"] == "Test Char"


# ---------------------------------------------------------------------------
# Qodo review follow-ups: bool coercion, path validation, format honesty
# ---------------------------------------------------------------------------


def test_character_book_string_booleans_parse_by_value():
    # bool("false") is True in Python - string flags must parse by value.
    book = {
        "entries": [
            {"keys": ["a"], "content": "x", "enabled": "false", "case_sensitive": "0"},
            {"keys": ["b"], "content": "y", "enabled": "true", "selective": "no", "constant": 0},
            {"keys": ["c"], "content": "z", "enabled": "yes", "constant": "on"},
            {"keys": ["d"], "content": "w", "enabled": "maybe"},  # unknown -> default True
            {"keys": ["e"], "content": "v", "enabled": 1, "case_sensitive": 0},
        ]
    }

    parsed = parse_character_book(book)
    a, b, c, d, e = parsed["entries"]

    assert a["enabled"] is False
    assert a["case_sensitive"] is False
    assert b["enabled"] is True
    assert b["selective"] is False
    assert b["constant"] is False
    assert c["enabled"] is True
    assert c["constant"] is True
    assert d["enabled"] is True  # unrecognized string falls back to the default
    assert e["enabled"] is True
    assert e["case_sensitive"] is False


def test_unsupported_image_extension_rejected_explicitly(tmp_path):
    # .jpg has no verified embedded-card extraction path; it must fail with a
    # clear unsupported-format path, not an obscure text-decode failure.
    jpg_path = tmp_path / "card.jpg"
    Image.new("RGB", (10, 10)).save(jpg_path, "JPEG")
    assert load_character_card_from_file(jpg_path) is None


def test_traversal_path_rejected():
    assert load_character_card_from_file("../../etc/passwd.json") is None
    assert load_character_card_from_file("..\\..\\Windows\\win.ini") is None
