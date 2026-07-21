"""P2f Task 3: proves the chat_events send-path union semantics (conversation
books + character-attached books, conversation wins on name collision) at the
resolver+processor level, without invoking the full send handler.

``_build_processor`` below is a byte-for-byte mirror of the union block wired
into ``chat_events.py`` (see the `enable_world_info` block, immediately before
the ``if has_character_book or world_books:`` init guard).
"""

from tldw_chatbook.Character_Chat.world_book_manager import (
    resolve_character_world_books,
)
from tldw_chatbook.Character_Chat.world_info_processor import WorldInfoProcessor


def _build_processor(conversation_books, char_data, has_native_book):
    # The exact shape of the chat_events union (union BEFORE the init guard).
    world_books = list(conversation_books)
    exclude_names = {str(b.get("name")) for b in world_books}
    world_books = world_books + resolve_character_world_books(char_data, exclude_names)
    if not (has_native_book or world_books):
        return None
    return WorldInfoProcessor(
        character_data=char_data if has_native_book else None,
        world_books=world_books if world_books else None,
    )


def _book(name, key, enabled=True):
    return {"name": name, "enabled": enabled,
            "entries": [{"keys": [key], "content": f"{name} lore", "enabled": True}]}


def test_attached_only_character_fires():
    # No conversation books, no native book: the attached book must still apply.
    char = {"extensions": {"character_world_books": [_book("Attached", "dragon")]}}
    proc = _build_processor([], char, has_native_book=False)
    assert proc is not None
    matched = proc.process_messages("a dragon appears", [])
    assert any("Attached lore" in e["content"] for e in matched["matched_entries"])


def test_conversation_wins_same_name():
    conv = [_book("World", "dragon")]
    char = {"extensions": {"character_world_books": [_book("World", "dragon")]}}
    proc = _build_processor(conv, char, has_native_book=False)
    fired = [e for e in proc.process_messages("dragon", [])["matched_entries"]]
    assert len(fired) == 1  # the character copy was excluded, only one fires


def test_disabled_attached_book_does_not_fire():
    char = {"extensions": {"character_world_books": [_book("Off", "dragon", enabled=False)]}}
    proc = _build_processor([], char, has_native_book=False)
    assert proc is None  # nothing to process


def test_no_attachment_is_noop():
    assert _build_processor([], {"extensions": {}}, has_native_book=False) is None
