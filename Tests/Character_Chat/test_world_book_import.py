import pytest

from tldw_chatbook.Character_Chat.world_book_import import normalize_world_book_import


def _tldw_entry(**kw):
    base = {"keys": ["Warden"], "content": "grim jailer", "position": "before_char",
            "insertion_order": 0, "selective": False, "secondary_keys": [],
            "case_sensitive": False, "enabled": True, "priority": 0}
    base.update(kw)
    return base


def test_tldw_export_passthrough():
    data = {"name": "B", "description": "d", "scan_depth": 3, "token_budget": 500,
            "recursive_scanning": False, "entries": [_tldw_entry(priority=42)]}
    out = normalize_world_book_import(data)
    assert out["name"] == "B" and out["scan_depth"] == 3
    e = out["entries"][0]
    assert e["keys"] == ["Warden"] and e["content"] == "grim jailer"
    assert e["priority"] == 42 and e["position"] == "before_char"


def test_character_book_array_passthrough():
    data = {"entries": [{"keys": ["a", "b"], "content": "c", "secondary_keys": ["s"],
                         "insertion_order": 2, "selective": True, "case_sensitive": True}]}
    e = normalize_world_book_import(data)["entries"][0]
    assert e["keys"] == ["a", "b"] and e["secondary_keys"] == ["s"]
    assert e["insertion_order"] == 2 and e["selective"] is True and e["case_sensitive"] is True


def test_sillytavern_world_info_object_form_remaps():
    data = {"entries": {"0": {"key": ["Warden"], "keysecondary": ["jail"], "content": "x",
                              "order": 5, "position": 1, "disable": True,
                              "caseSensitive": True, "selective": True}}}
    e = normalize_world_book_import(data)["entries"][0]
    assert e["keys"] == ["Warden"] and e["secondary_keys"] == ["jail"]
    assert e["insertion_order"] == 5 and e["position"] == "after_char"
    assert e["enabled"] is False and e["case_sensitive"] is True and e["selective"] is True


def test_missing_entries_yields_empty_list():
    assert normalize_world_book_import({"name": "B"})["entries"] == []


def test_priority_and_extensions_preserved_and_defaulted():
    data = {"entries": [{"keys": ["k"], "content": "c", "extensions": {"x": 1}},
                        {"keys": ["k2"], "content": "c2"}]}
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
    e = normalize_world_book_import({"entries": [{"keys": ["Warden", None], "content": "c"}]})["entries"][0]
    assert e["keys"] == ["Warden"]


def test_keys_of_only_nulls_is_rejected():
    """A keys list of only nulls has no real keys → rejected (not ["None"])."""
    with pytest.raises(ValueError, match="Entry 1 has no keys"):
        normalize_world_book_import({"entries": [{"keys": [None], "content": "c"}]})


def test_bool_position_does_not_alias_to_int_position():
    """bool is an int subclass; the isinstance(pos, bool) guard must keep
    position True/False from aliasing to the int-position map (1/0)."""
    e_true = normalize_world_book_import({"entries": [{"keys": ["k"], "content": "c", "position": True}]})["entries"][0]
    e_false = normalize_world_book_import({"entries": [{"keys": ["k"], "content": "c", "position": False}]})["entries"][0]
    assert e_true["position"] == "before_char" and e_false["position"] == "before_char"


def test_unknown_int_position_defaults_before_char():
    e = normalize_world_book_import({"entries": [{"keys": ["k"], "content": "c", "position": 7}]})["entries"][0]
    assert e["position"] == "before_char"


def test_bare_string_key_becomes_single_item_list():
    e = normalize_world_book_import({"entries": [{"key": "Warden", "content": "c"}]})["entries"][0]
    assert e["keys"] == ["Warden"]


def test_non_dict_extensions_defaults_empty():
    e = normalize_world_book_import({"entries": [{"keys": ["k"], "content": "c", "extensions": "oops"}]})["entries"][0]
    assert e["extensions"] == {}


def test_non_numeric_order_falls_back_to_index():
    entries = normalize_world_book_import({"entries": [
        {"keys": ["a"], "content": "c"},
        {"keys": ["b"], "content": "c", "order": "abc"},
    ]})["entries"]
    assert entries[1]["insertion_order"] == 1


def test_explicit_enabled_false_preserved():
    e = normalize_world_book_import({"entries": [{"keys": ["k"], "content": "c", "enabled": False}]})["entries"][0]
    assert e["enabled"] is False
