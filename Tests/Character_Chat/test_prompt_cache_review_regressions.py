"""Real-service regressions from the independent PR3045 review."""

from threading import Event, Thread

import pytest

from tldw_chatbook.Character_Chat import Chat_Dictionary_Lib as dictionaries
from tldw_chatbook.Character_Chat import world_info_resolver as resolver
from tldw_chatbook.Character_Chat.local_character_persona_service import (
    LocalCharacterPersonaService,
)
from tldw_chatbook.Character_Chat.local_chat_dictionary_service import (
    LocalChatDictionaryService,
)
from tldw_chatbook.Character_Chat.world_book_manager import WorldBookManager
from tldw_chatbook.Character_Chat.world_info_processor import WorldInfoProcessor
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def db(tmp_path):
    value = CharactersRAGDB(tmp_path / "prompt-cache.db", "review-test")
    yield value
    value.close_connection()


@pytest.fixture(autouse=True)
def isolated_caches():
    resolver._clear_world_info_cache()
    dictionaries._clear_dictionary_bundle_cache()
    yield
    resolver._clear_world_info_cache()
    dictionaries._clear_dictionary_bundle_cache()


def _extensions(content):
    return {
        "character_book": {
            "name": "Native",
            "entries": [{"keys": ["castle"], "content": content}],
        },
        "chat_dictionaries": [
            {
                "name": "Native",
                "enabled": True,
                "entries": [{"key": "castle", "content": content}],
            }
        ],
    }


@pytest.fixture(params=["world-info", "dictionary"])
def inject(request):
    if request.param == "world-info":
        return lambda db, conversation, card: resolver.apply_world_info_to_message(
            db, conversation, card, "castle", []
        )
    return lambda db, conversation, card: dictionaries.apply_active_chatdicts_to_text(
        db, conversation, card, "castle"
    )


@pytest.mark.parametrize("writer", ["persona", "ccp"])
def test_native_card_save_refreshes_warm_prompt_content(
    db, inject, monkeypatch, writer
):
    conversation = db.add_conversation({"title": "Review"})
    character = db.add_character_card(
        {"name": "Review", "extensions": _extensions("OLD")}
    )
    card = db.get_character_card_by_id(character)
    assert "OLD" in inject(db, conversation, card)
    assert "OLD" in inject(db, conversation, card)

    if writer == "persona":
        LocalCharacterPersonaService(db).update_character(
            character,
            {"extensions": _extensions("NEW")},
            expected_version=card["version"],
        )
    else:
        from tldw_chatbook.UI.CCP_Modules import ccp_character_handler

        monkeypatch.setattr(ccp_character_handler, "_default_character_db", lambda: db)
        assert ccp_character_handler.update_character(
            str(character), {"extensions": _extensions("NEW")}
        )
    updated = db.get_character_card_by_id(character)
    assert "NEW" in inject(db, conversation, updated)
    assert "NEW" in inject(db, conversation, updated)


def test_unversioned_card_content_cannot_freeze_a_warm_prompt(db, inject):
    conversation = db.add_conversation({"title": "Review"})
    card = {"id": 7, "extensions": _extensions("OLD")}
    assert "OLD" in inject(db, conversation, card)
    card["extensions"] = _extensions("NEW")
    assert "NEW" in inject(db, conversation, card)


def _world_store(db):
    conversation = db.add_conversation({"title": "Review"})
    manager = WorldBookManager(db)
    book = manager.create_world_book("Review")
    entry = manager.create_world_book_entry(book, keys=["castle"], content="OLD")
    manager.associate_world_book_with_conversation(conversation, book)
    return conversation, manager, entry


def _world_text(db, conversation):
    return resolver.apply_world_info_to_message(db, conversation, None, "castle", [])


@pytest.mark.parametrize("transaction", ["managed", "nested", "borrowed"])
def test_reader_during_book_write_refreshes_after_commit(db, monkeypatch, transaction):
    conversation, manager, entry = _world_store(db)
    assert "OLD" in _world_text(db, conversation)
    written = Event()
    release = Event()
    errors = []

    if transaction == "managed":
        original = manager._bump_generation

        def pause_after_invalidation():
            original()
            written.set()
            assert release.wait(10)

        monkeypatch.setattr(manager, "_bump_generation", pause_after_invalidation)

    def writer():
        try:
            if transaction == "managed":
                assert manager.update_world_book_entry(entry, content="NEW")
            elif transaction == "nested":
                with db.transaction():
                    assert manager.update_world_book_entry(entry, content="NEW")
                    written.set()
                    assert release.wait(10)
            else:
                connection = db.get_connection()
                connection.execute("BEGIN IMMEDIATE")
                assert manager.update_world_book_entry(entry, content="NEW")
                written.set()
                assert release.wait(10)
                connection.commit()
        except BaseException as error:  # noqa: BLE001 - report worker failures to the test thread
            errors.append(error)
        finally:
            db.close_connection()

    thread = Thread(target=writer)
    thread.start()
    try:
        assert written.wait(10)
        assert "OLD" in _world_text(db, conversation)
        release.set()
        thread.join(10)
        assert not thread.is_alive()
        assert not errors
        assert "NEW" in _world_text(db, conversation)
        assert "NEW" in _world_text(db, conversation)
    finally:
        release.set()
        thread.join(10)


@pytest.mark.parametrize("transaction", ["managed", "borrowed"])
def test_rolled_back_book_content_never_escapes_into_cache(db, transaction):
    conversation, manager, entry = _world_store(db)
    assert "OLD" in _world_text(db, conversation)
    if transaction == "managed":
        with pytest.raises(RuntimeError, match="rollback"), db.transaction():
            manager.update_world_book_entry(entry, content="NEW")
            assert "NEW" in _world_text(db, conversation)
            raise RuntimeError("rollback")
    else:
        connection = db.get_connection()
        connection.execute("BEGIN IMMEDIATE")
        try:
            manager.update_world_book_entry(entry, content="NEW")
            assert "NEW" in _world_text(db, conversation)
        finally:
            connection.rollback()
    assert "OLD" in _world_text(db, conversation)


def test_ordinary_chat_writes_preserve_warm_world_processor(db, monkeypatch):
    conversation, _manager, _entry = _world_store(db)
    fetches = []
    original = WorldBookManager.get_world_books_for_conversation

    def count_fetch(self, *args, **kwargs):
        fetches.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(
        WorldBookManager, "get_world_books_for_conversation", count_fetch
    )
    assert "OLD" in _world_text(db, conversation)
    db.add_message(
        {
            "conversation_id": conversation,
            "sender": "User",
            "role": "user",
            "content": "castle",
        }
    )
    assert "OLD" in _world_text(db, conversation)
    assert len(fetches) == 1


def _dictionary_store(db):
    conversation = db.add_conversation({"title": "Review"})
    old = dictionaries.save_chat_dictionary(
        db, "Old", entries=[dictionaries.ChatDictionary(key="castle", content="OLD")]
    )
    new = dictionaries.save_chat_dictionary(
        db, "New", entries=[dictionaries.ChatDictionary(key="castle", content="NEW")]
    )
    service = LocalChatDictionaryService(db)
    service.attach_to_conversation(old, conversation)

    def update():
        service.detach_from_conversation(old, conversation)
        service.attach_to_conversation(new, conversation)

    def read():
        return dictionaries.apply_active_chatdicts_to_text(
            db, conversation, None, "castle"
        )

    return update, read


@pytest.mark.parametrize("transaction", ["managed", "borrowed"])
def test_dictionary_attachment_refreshes_after_outer_commit(db, transaction):
    update, read = _dictionary_store(db)
    assert read() == "OLD"
    written = Event()
    release = Event()
    errors = []

    def writer():
        try:
            if transaction == "managed":
                with db.transaction():
                    update()
                    written.set()
                    assert release.wait(10)
            else:
                connection = db.get_connection()
                connection.execute("BEGIN IMMEDIATE")
                update()
                written.set()
                assert release.wait(10)
                connection.commit()
        except BaseException as error:  # noqa: BLE001 - report worker failures to the test thread
            errors.append(error)
        finally:
            db.close_connection()

    thread = Thread(target=writer)
    thread.start()
    try:
        assert written.wait(10)
        assert read() == "OLD"
        release.set()
        thread.join(10)
        assert not thread.is_alive()
        assert not errors
        assert read() == "NEW"
        assert read() == "NEW"
    finally:
        release.set()
        thread.join(10)


@pytest.mark.parametrize("transaction", ["managed", "borrowed"])
def test_rolled_back_dictionary_attachment_never_escapes_into_cache(db, transaction):
    update, read = _dictionary_store(db)
    assert read() == "OLD"
    if transaction == "managed":
        with pytest.raises(RuntimeError, match="rollback"), db.transaction():
            update()
            assert read() == "NEW"
            raise RuntimeError("rollback")
    else:
        connection = db.get_connection()
        connection.execute("BEGIN IMMEDIATE")
        try:
            update()
            assert read() == "NEW"
        finally:
            connection.rollback()
    assert read() == "OLD"


def test_recursively_activated_equal_lore_is_injected_once():
    duplicate = {
        "keys": ["dragon"],
        "content": "A dragon lore fact",
        "insertion_order": 2,
    }
    book = {
        "recursive_scanning": True,
        "token_budget": 100000,
        "entries": [
            {
                "keys": ["castle"],
                "content": "A castle has a dragon",
                "insertion_order": 1,
            },
            dict(duplicate),
            dict(duplicate),
        ],
    }
    result = WorldInfoProcessor(world_books=[book]).process_messages("castle", [])
    assert [row["content"] for row in result["matched_entries"]] == [
        "A castle has a dragon",
        "A dragon lore fact",
    ]
