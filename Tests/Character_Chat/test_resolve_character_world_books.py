from tldw_chatbook.Character_Chat.world_book_manager import (
    resolve_character_world_books,
)
from tldw_chatbook.Character_Chat.world_info_processor import WorldInfoProcessor


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


def test_malformed_scalar_fields_are_sanitized():
    blocks = [
        {
            "name": "Bad",
            "enabled": True,
            "scan_depth": "high",
            "token_budget": None,
            "priority": "x",
            "entries": "notalist",
        }
    ]
    out = resolve_character_world_books(_char(blocks), set())
    assert len(out) == 1
    book = out[0]
    assert book["scan_depth"] == 3
    assert book["token_budget"] == 500
    assert book["priority"] == 0
    assert book["entries"] == []

    # Integration: the sanitized block must never make the processor choke
    # during init (this is the whole point of the fix — a malformed embedded
    # snapshot must not silently disable world-info injection for the send).
    processor = WorldInfoProcessor(character_data=None, world_books=out)
    assert processor.scan_depth == 3
    assert processor.token_budget == 500


def test_malformed_entries_drop_non_dict_items():
    blocks = [
        {
            "name": "Mixed",
            "enabled": True,
            "entries": [{"keys": ["k"], "content": "c"}, "junk", None],
        }
    ]
    out = resolve_character_world_books(_char(blocks), set())
    assert len(out) == 1
    assert out[0]["entries"] == [{"keys": ["k"], "content": "c"}]
