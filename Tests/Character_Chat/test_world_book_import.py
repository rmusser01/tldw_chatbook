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
