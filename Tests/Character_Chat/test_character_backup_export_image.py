# test_character_backup_export_image.py
"""task-15769: the character JSON backup dump must survive image-bearing cards.

`Tools_Settings_Window._export_characters_worker` serializes raw
`list_character_cards` rows. Two row-level values are not JSON-serializable:

- the `image` BLOB (``bytes``) whenever a card has an avatar, and
- `created_at`/`last_modified` (``datetime``) on EVERY row,

so the backup crashed with ``TypeError: Object of type ... is not JSON
serializable``. These tests pin the serialization helper the worker now
uses: raw BLOB replaced by a plain-base64 ``image_base64`` string (the
`Chat_Functions.load_characters` compatibility shape -- deliberately NOT a
data-URI, because the import chain b64decodes the raw string), datetimes
as ISO strings, and a full export -> re-import round trip that restores the
image byte-for-byte.
"""

import base64
import json

import pytest

from tldw_chatbook.Character_Chat.Character_Chat_Lib import (
    import_and_save_character_from_file,
    import_character_card_from_json_string,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Tools_Settings_Window import (
    _serialize_character_cards_for_backup,
)

pytestmark = pytest.mark.integration

# Not a decodable PNG on purpose: the export must round-trip the BLOB
# byte-for-byte without caring what's inside it. Covers all 256 byte values.
IMAGE_BYTES = b"\x89PNG\r\n\x1a\n" + bytes(range(256))


@pytest.fixture
def db_instance(tmp_path):
    db = CharactersRAGDB(tmp_path / "backup_export.sqlite", "test_client")
    yield db
    db.close_connection()


@pytest.fixture
def backup_rows(db_instance):
    """One image-bearing card + one imageless card, as the worker reads them."""
    db_instance.add_character_card(
        {
            "name": "Image Bearer",
            "description": "card with an avatar BLOB",
            "first_message": "Hello.",
            "tags": ["visual"],
            "image": IMAGE_BYTES,
        }
    )
    db_instance.add_character_card(
        {
            "name": "Plain Card",
            "description": "card without an avatar",
        }
    )
    # The worker's exact projection: a backup must include the image BLOBs.
    return db_instance.list_character_cards(limit=10000, include_image=True)


def _exported_by_name(rows):
    exported = json.loads(_serialize_character_cards_for_backup(rows))
    return {card["name"]: card for card in exported}


def test_backup_export_of_image_bearing_cards_is_valid_json(backup_rows):
    """The crash path: an image-bearing row must not kill the whole dump."""
    assert any(isinstance(row.get("image"), bytes) for row in backup_rows), (
        "fixture must actually contain an image BLOB row"
    )

    json_str = _serialize_character_cards_for_backup(backup_rows)

    by_name = {card["name"]: card for card in json.loads(json_str)}
    card = by_name["Image Bearer"]
    # The raw BLOB never reaches the JSON; the image is carried as plain
    # base64 under the load_characters-compatible key.
    assert "image" not in card
    assert base64.b64decode(card["image_base64"]) == IMAGE_BYTES


def test_backup_export_serializes_datetime_rows(backup_rows):
    """Every row carries datetime created_at/last_modified; the dump must not crash."""
    by_name = _exported_by_name(backup_rows)
    for card in by_name.values():
        assert isinstance(card["created_at"], str)
        assert isinstance(card["last_modified"], str)


def test_backup_export_imageless_card_carries_no_image_keys(backup_rows):
    by_name = _exported_by_name(backup_rows)
    card = by_name["Plain Card"]
    assert "image" not in card
    assert "image_base64" not in card


def test_exported_card_reimports_with_identical_image_bytes(backup_rows, tmp_path):
    """Full round trip: export -> card file -> import into a fresh DB -> same bytes."""
    by_name = _exported_by_name(backup_rows)
    card_path = tmp_path / "image_bearer.json"
    card_path.write_text(
        json.dumps(by_name["Image Bearer"], ensure_ascii=False), encoding="utf-8"
    )

    reimport_db = CharactersRAGDB(tmp_path / "reimport.sqlite", "reimport_client")
    try:
        char_id = import_and_save_character_from_file(reimport_db, str(card_path))
        assert char_id is not None
        restored = reimport_db.get_character_card_by_id(char_id)
        assert restored["image"] == IMAGE_BYTES
        # Bloat guard: the base64 payload must be consumed as the image, not
        # duplicated wholesale into the card's extensions JSON.
        assert "image_base64" not in (restored.get("extensions") or {})
    finally:
        reimport_db.close_connection()


def test_parse_path_accepts_image_base64_key_directly():
    """parse_v1_card must treat image_base64 as an image source, not an extension."""
    payload = base64.b64encode(IMAGE_BYTES).decode("ascii")
    parsed = import_character_card_from_json_string(
        json.dumps({"name": "Inline", "image_base64": payload})
    )
    assert parsed is not None
    assert parsed["image_base64"] == payload
    assert "image_base64" not in parsed["extensions"]
