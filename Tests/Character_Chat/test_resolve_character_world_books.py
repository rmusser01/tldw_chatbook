from tldw_chatbook.Character_Chat.world_book_manager import (
    resolve_character_world_books,
)


def _char(blocks):
    return {"extensions": {"character_world_books": blocks}}


def test_returns_enabled_attached_books():
    blocks = [{"name": "A", "enabled": True, "entries": [{"keys": ["x"], "content": "c"}]}]
    out = resolve_character_world_books(_char(blocks), set())
    assert [b["name"] for b in out] == ["A"]


def test_conversation_wins_by_name():
    blocks = [{"name": "Shared", "enabled": True, "entries": []},
              {"name": "Solo", "enabled": True, "entries": []}]
    out = resolve_character_world_books(_char(blocks), {"Shared"})
    assert [b["name"] for b in out] == ["Solo"]


def test_disabled_book_dropped():
    blocks = [{"name": "Off", "enabled": False, "entries": []}]
    assert resolve_character_world_books(_char(blocks), set()) == []


def test_string_false_enabled_is_falsey():
    blocks = [{"name": "Off", "enabled": "false", "entries": []}]
    assert resolve_character_world_books(_char(blocks), set()) == []


def test_dedup_by_name_first_wins():
    blocks = [{"name": "Dup", "enabled": True, "entries": [{"keys": ["1"], "content": "a"}]},
              {"name": "Dup", "enabled": True, "entries": [{"keys": ["2"], "content": "b"}]}]
    out = resolve_character_world_books(_char(blocks), set())
    assert len(out) == 1 and out[0]["entries"][0]["keys"] == ["1"]


def test_malformed_inputs_never_raise():
    assert resolve_character_world_books(None, set()) == []
    assert resolve_character_world_books({}, set()) == []
    assert resolve_character_world_books({"extensions": "not-a-dict"}, set()) == []
    assert resolve_character_world_books({"extensions": {"character_world_books": "x"}}, set()) == []
    assert resolve_character_world_books({"extensions": {"character_world_books": [None, 3, {"no": "name"}]}}, set()) == []
