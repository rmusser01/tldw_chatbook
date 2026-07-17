"""P1f AC6: a character's embedded chat_dictionaries survive export → import.

Exercises the *existing* export/import machinery (no P1f code change to that
path) as a characterization guard that the portable-snapshot promise the P1f
spec's AC6 relies on actually holds: an ``extensions['chat_dictionaries']``
block written by ``LocalChatDictionaryService.attach_to_character`` must ride
through ``export_character_card_to_json`` -> ``import_character_card_from_json_string``
intact.

Real-shape notes (verified by running this test against the live code, not
assumed from the spec):

- ``export_character_card_to_json`` returns a JSON **string** (a
  ``json.dumps`` of a ``{"spec": "chara_card_v2", "data": {...}}`` envelope),
  not a dict - the ``isinstance(exported, str)`` guard from the brief is a
  no-op here but is kept for robustness.
- ``import_character_card_from_json_string`` returns a **flat** dict (DB
  schema shape - ``parse_v2_card`` copies ``data['extensions']`` straight to
  the top-level ``extensions`` key); it is NOT nested under a ``data`` key on
  the imported side.
- ``chara_card_v2`` validation (``validate_v2_card``) requires ``name``,
  ``description``, ``personality``, ``scenario``, ``first_mes`` and
  ``mes_example`` to all be present *and be strings* in the exported
  ``data`` node, with ``name``/``first_mes`` additionally non-blank.
  ``export_character_card_to_json`` passes DB column values through with
  only a partial ``... or ''`` guard (``scenario``/``first_mes`` only, not
  ``description``/``personality``), so a character created with only
  ``{"name": ...}`` round-trips DB ``None`` straight into those fields and
  V2 validation - and therefore the whole import - fails. This is a
  pre-existing quirk of the general export/import path unrelated to P1f, so
  the fixture below supplies real string values for every required V2 field
  (matching the pattern already used by
  ``Tests/Character_Chat/test_character_export_no_image.py``) to isolate the
  assertion under test: does the embedded dictionary block survive.
"""

import json

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Character_Chat.local_chat_dictionary_service import LocalChatDictionaryService
from tldw_chatbook.Character_Chat.Character_Chat_Lib import (
    export_character_card_to_json,
    import_character_card_from_json_string,
)


def test_embedded_chat_dictionaries_survive_export_import(tmp_path):
    db = CharactersRAGDB(tmp_path / "port.db", "test-client")
    service = LocalChatDictionaryService(db)
    dict_id = service.create_dictionary(
        {"name": "Slang", "entries": [{"pattern": "hi", "replacement": "hello"}]}
    )["id"]
    char_id = db.add_character_card(
        {
            "name": "Noir",
            "description": "A stub character for the portability test.",
            "personality": "Stoic.",
            "scenario": "A rain-soaked city.",
            "first_message": "The city never sleeps, and neither do I.",
        }
    )
    service.attach_to_character(dict_id, char_id)

    exported = export_character_card_to_json(db, char_id, include_image=False)
    assert exported is not None, "export must succeed for a character with all required V2 fields"
    payload = exported if isinstance(exported, str) else json.dumps(exported)
    imported = import_character_card_from_json_string(payload)
    assert imported is not None, "import must succeed on the card this export just produced"

    ext = (imported or {}).get("extensions") or {}
    names = [b.get("name") for b in (ext.get("chat_dictionaries") or [])]
    assert "Slang" in names, "embedded chat_dictionaries must ride inside extensions across export/import"
