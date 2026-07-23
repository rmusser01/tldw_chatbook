import pytest

from tldw_chatbook.Character_Chat.world_book_import import (
    normalize_world_book_import,
    character_book_to_world_book_block,
)


def _tldw_entry(**kw):
    base = {
        "keys": ["Warden"],
        "content": "grim jailer",
        "position": "before_char",
        "insertion_order": 0,
        "selective": False,
        "secondary_keys": [],
        "case_sensitive": False,
        "enabled": True,
        "priority": 0,
    }
    base.update(kw)
    return base


def test_tldw_export_passthrough():
    data = {
        "name": "B",
        "description": "d",
        "scan_depth": 3,
        "token_budget": 500,
        "recursive_scanning": False,
        "entries": [_tldw_entry(priority=42)],
    }
    out = normalize_world_book_import(data)
    assert out["name"] == "B" and out["scan_depth"] == 3
    e = out["entries"][0]
    assert e["keys"] == ["Warden"] and e["content"] == "grim jailer"
    assert e["priority"] == 42 and e["position"] == "before_char"


def test_character_book_array_passthrough():
    data = {
        "entries": [
            {
                "keys": ["a", "b"],
                "content": "c",
                "secondary_keys": ["s"],
                "insertion_order": 2,
                "selective": True,
                "case_sensitive": True,
            }
        ]
    }
    e = normalize_world_book_import(data)["entries"][0]
    assert e["keys"] == ["a", "b"] and e["secondary_keys"] == ["s"]
    assert (
        e["insertion_order"] == 2
        and e["selective"] is True
        and e["case_sensitive"] is True
    )


def test_sillytavern_world_info_object_form_remaps():
    data = {
        "entries": {
            "0": {
                "key": ["Warden"],
                "keysecondary": ["jail"],
                "content": "x",
                "order": 5,
                "position": 1,
                "disable": True,
                "caseSensitive": True,
                "selective": True,
            }
        }
    }
    e = normalize_world_book_import(data)["entries"][0]
    assert e["keys"] == ["Warden"] and e["secondary_keys"] == ["jail"]
    assert e["insertion_order"] == 5 and e["position"] == "after_char"
    assert (
        e["enabled"] is False and e["case_sensitive"] is True and e["selective"] is True
    )


def test_missing_entries_yields_empty_list():
    assert normalize_world_book_import({"name": "B"})["entries"] == []


def test_priority_and_extensions_preserved_and_defaulted():
    data = {
        "entries": [
            {"keys": ["k"], "content": "c", "extensions": {"x": 1}},
            {"keys": ["k2"], "content": "c2"},
        ]
    }
    out = normalize_world_book_import(data)["entries"]
    assert out[0]["extensions"] == {"x": 1} and out[0]["priority"] == 0
    assert out[1]["extensions"] == {} and out[1]["insertion_order"] == 1


def test_non_dict_top_level_raises():
    with pytest.raises(ValueError, match="must be a JSON object"):
        normalize_world_book_import([1, 2, 3])


def test_entries_not_list_or_dict_raises():
    with pytest.raises(ValueError, match="must be a list or an object"):
        normalize_world_book_import({"entries": 42})


def test_empty_keys_entry_raises():
    with pytest.raises(ValueError, match="Entry 1 has no keys"):
        normalize_world_book_import({"entries": [{"keys": [], "content": "c"}]})


def test_empty_content_entry_raises():
    with pytest.raises(ValueError, match="Entry 1 has no content"):
        normalize_world_book_import({"entries": [{"keys": ["k"], "content": "   "}]})


def test_non_dict_entry_raises():
    with pytest.raises(ValueError, match="Entry 1 is not an object"):
        normalize_world_book_import({"entries": ["not a dict"]})


# --- whole-branch-review Minor coverage (M1 null-in-keys, M2 branch guards) ---


def test_null_in_keys_list_is_dropped_not_stringified():
    """A null inside a keys array is dropped, not turned into the literal "None"."""
    e = normalize_world_book_import(
        {"entries": [{"keys": ["Warden", None], "content": "c"}]}
    )["entries"][0]
    assert e["keys"] == ["Warden"]


def test_keys_of_only_nulls_is_rejected():
    """A keys list of only nulls has no real keys → rejected (not ["None"])."""
    with pytest.raises(ValueError, match="Entry 1 has no keys"):
        normalize_world_book_import({"entries": [{"keys": [None], "content": "c"}]})


def test_bool_position_does_not_alias_to_int_position():
    """bool is an int subclass; the isinstance(pos, bool) guard must keep
    position True/False from aliasing to the int-position map (1/0)."""
    e_true = normalize_world_book_import(
        {"entries": [{"keys": ["k"], "content": "c", "position": True}]}
    )["entries"][0]
    e_false = normalize_world_book_import(
        {"entries": [{"keys": ["k"], "content": "c", "position": False}]}
    )["entries"][0]
    assert e_true["position"] == "before_char" and e_false["position"] == "before_char"


def test_unknown_int_position_defaults_before_char():
    e = normalize_world_book_import(
        {"entries": [{"keys": ["k"], "content": "c", "position": 7}]}
    )["entries"][0]
    assert e["position"] == "before_char"


def test_bare_string_key_becomes_single_item_list():
    e = normalize_world_book_import({"entries": [{"key": "Warden", "content": "c"}]})[
        "entries"
    ][0]
    assert e["keys"] == ["Warden"]


def test_non_dict_extensions_defaults_empty():
    e = normalize_world_book_import(
        {"entries": [{"keys": ["k"], "content": "c", "extensions": "oops"}]}
    )["entries"][0]
    assert e["extensions"] == {}


def test_non_numeric_order_falls_back_to_index():
    entries = normalize_world_book_import(
        {
            "entries": [
                {"keys": ["a"], "content": "c"},
                {"keys": ["b"], "content": "c", "order": "abc"},
            ]
        }
    )["entries"]
    assert entries[1]["insertion_order"] == 1


def test_explicit_enabled_false_preserved():
    e = normalize_world_book_import(
        {"entries": [{"keys": ["k"], "content": "c", "enabled": False}]}
    )["entries"][0]
    assert e["enabled"] is False


def test_null_content_is_rejected_not_stringified():
    """Explicit null content must raise (not become the literal "None") — Gemini
    #701 high-severity finding, parallel to the null-in-keys trap."""
    with pytest.raises(ValueError, match="Entry 1 has no content"):
        normalize_world_book_import({"entries": [{"keys": ["k"], "content": None}]})


# --- Qodo #701 finding 4: robust boolean coercion for loosely-typed files ---


def _b(**kw):
    return normalize_world_book_import(
        {"entries": [{"keys": ["k"], "content": "c", **kw}]}
    )["entries"][0]


def test_string_false_booleans_are_not_truthy():
    """bool("false") is True in Python — the coercer must map string booleans."""
    e = _b(selective="false", case_sensitive="false", enabled="false")
    assert (
        e["selective"] is False
        and e["case_sensitive"] is False
        and e["enabled"] is False
    )


def test_string_true_booleans_map_true():
    e = _b(selective="true", caseSensitive="TRUE")
    assert e["selective"] is True and e["case_sensitive"] is True


def test_null_enabled_defaults_true():
    assert _b(enabled=None)["enabled"] is True


def test_string_disable_false_means_enabled():
    """SillyTavern `disable: "false"` (string) must mean enabled=True."""
    assert _b(disable="false")["enabled"] is True


def test_disable_true_flag_disables_when_no_enabled():
    assert _b(disable=True)["enabled"] is False


def test_regex_flag_normalized():
    e = normalize_world_book_import({"entries": [{"keys": ["w[ao]rden"], "content": "c", "regex": True}]})["entries"][0]
    assert e["regex"] is True


def test_bad_regex_pattern_rejects_file():
    with pytest.raises(ValueError, match="Entry 1"):
        normalize_world_book_import({"entries": [{"keys": ["(a+)+"], "content": "c", "regex": True}]})


def test_bad_pattern_ignored_when_not_regex():
    # A would-be-bad "pattern" in a non-regex entry is a literal keyword — never validated.
    e = normalize_world_book_import({"entries": [{"keys": ["(a+)+"], "content": "c"}]})["entries"][0]
    assert e["keys"] == ["(a+)+"] and e["regex"] is False


# --- character_book_to_world_book_block (task-429) ---


def test_character_book_to_block_basic():
    book = {
        "name": "Second Chance Lore",
        "description": "ship lore",
        "scan_depth": 5,
        "token_budget": 300,
        "recursive_scanning": True,
        "entries": [
            {"keys": ["coffee"], "content": "The machine explodes.",
             "enabled": True, "insertion_order": 1, "position": 0},
            {"keys": ["airlock"], "content": "It sticks.",
             "enabled": True, "insertion_order": 2},
        ],
    }
    block, imported, skipped = character_book_to_world_book_block(book, "X Lorebook")
    assert imported == 2 and skipped == 0
    assert block["name"] == "Second Chance Lore"
    assert block["scan_depth"] == 5 and block["token_budget"] == 300
    assert block["recursive_scanning"] is True and block["enabled"] is True
    # int position (0) normalized to the string enum
    assert block["entries"][0]["position"] == "before_char"
    assert block["entries"][0]["keys"] == ["coffee"]
    assert block["entries"][0]["regex"] is False


def test_character_book_to_block_skips_unsalvageable_and_counts():
    book = {"name": "B", "entries": [
        {"keys": ["ok"], "content": "good", "enabled": True, "insertion_order": 1},
        {"content": "no keys", "enabled": True, "insertion_order": 2},   # no keys -> skip
        {"keys": ["x"], "enabled": True, "insertion_order": 3},          # no content -> skip
    ]}
    block, imported, skipped = character_book_to_world_book_block(book, "X Lorebook")
    assert imported == 1 and skipped == 2
    assert len(block["entries"]) == 1


def test_character_book_to_block_empty_name_uses_fallback():
    block, _, _ = character_book_to_world_book_block(
        {"name": "", "entries": []}, "Elara Lorebook")
    assert block["name"] == "Elara Lorebook"


def test_character_book_to_block_non_dict_returns_none():
    assert character_book_to_world_book_block(None, "X") == (None, 0, 0)
    assert character_book_to_world_book_block([1, 2], "X") == (None, 0, 0)


def test_character_book_to_block_entries_as_object_form():
    book = {"name": "B", "entries": {"0": {"keys": ["k"], "content": "c",
            "enabled": True, "insertion_order": 1}}}
    block, imported, skipped = character_book_to_world_book_block(book, "X")
    assert imported == 1 and block["entries"][0]["keys"] == ["k"]
