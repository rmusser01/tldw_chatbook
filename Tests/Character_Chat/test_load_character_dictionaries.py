"""P1f: defensive parse of a character card's embedded chat dictionaries."""

from tldw_chatbook.Character_Chat.Chat_Dictionary_Lib import load_character_dictionaries, ChatDictionary


def test_parses_embedded_blocks_into_chatdictionary_entries():
    char = {"extensions": {"chat_dictionaries": [
        {"name": "Slang", "enabled": True, "entries": [
            {"key": "BP", "content": "blood pressure", "probability": 100},
        ]},
    ]}}
    blocks = load_character_dictionaries(char)
    assert len(blocks) == 1
    assert blocks[0]["name"] == "Slang"
    assert blocks[0]["enabled"] is True
    assert len(blocks[0]["entries"]) == 1
    assert isinstance(blocks[0]["entries"][0], ChatDictionary)


def test_skips_malformed_blocks_and_entries_without_raising():
    char = {"extensions": {"chat_dictionaries": [
        "not-a-dict",
        {"name": "", "entries": []},                 # no name → skipped
        {"name": "Bad", "entries": [{"content": "x"}]},  # entry missing 'key' → entry skipped
        {"name": "Good", "entries": [{"key": "k", "content": "c"}]},
    ]}}
    blocks = load_character_dictionaries(char)
    names = [b["name"] for b in blocks]
    assert names == ["Bad", "Good"]
    assert load_character_dictionaries(char)[0]["entries"] == []  # 'Bad' dropped its bad entry


def test_tolerates_none_and_missing_and_string_extensions():
    assert load_character_dictionaries(None) == []
    assert load_character_dictionaries({}) == []
    assert load_character_dictionaries({"extensions": {}}) == []
    assert load_character_dictionaries({"extensions": '{"chat_dictionaries": []}'}) == []
    assert load_character_dictionaries({"extensions": "not json"}) == []


def test_non_list_entries_is_tolerated_without_raising():
    char = {"extensions": {"chat_dictionaries": [{"name": "Evil", "entries": 5}]}}
    blocks = load_character_dictionaries(char)
    assert blocks == [{"name": "Evil", "enabled": True, "entries": []}]


def test_enabled_false_string_is_coerced_honestly():
    """A malformed imported snapshot can carry enabled as the string "false".
    bool("false") is True (truthy-string bug), so it must be honestly parsed
    (via _coerce_bool) instead of applied as enabled."""
    char = {"extensions": {"chat_dictionaries": [
        {"name": "Off", "enabled": "false", "entries": [{"key": "k", "content": "c"}]},
    ]}}
    blocks = load_character_dictionaries(char)
    assert blocks[0]["enabled"] is False


def test_duplicate_named_blocks_dedup_to_first_occurrence():
    """A hostile/crafted card can embed two blocks with the same name.

    ``attach_to_character`` dedups by name so it can never create this, but a
    crafted import can. Downstream (``collect_active_chatdict_entries`` and
    the character-dictionaries panel) must never see two same-named blocks:
    the first occurrence wins.
    """
    char = {"extensions": {"chat_dictionaries": [
        {"name": "Dup", "entries": [{"key": "a", "content": "1"}]},
        {"name": "Dup", "entries": [{"key": "b", "content": "2"}]},
    ]}}
    blocks = load_character_dictionaries(char)
    assert len(blocks) == 1
    assert blocks[0]["name"] == "Dup"
    assert len(blocks[0]["entries"]) == 1
    assert blocks[0]["entries"][0].key == "a"
