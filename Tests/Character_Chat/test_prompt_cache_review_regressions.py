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

        def fn(db, conversation, card):
            return resolver.apply_world_info_to_message(
                db, conversation, card, "castle", []
            )

    else:

        def fn(db, conversation, card):
            return dictionaries.apply_active_chatdicts_to_text(
                db, conversation, card, "castle"
            )

    fn.kind = request.param  # lets tests branch per cache without cross-products
    return fn


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


def _save_card(db, monkeypatch, writer, character, card, payload):
    """Write a card update through the named in-app seam and re-read the card."""
    if writer == "persona":
        LocalCharacterPersonaService(db).update_character(
            character, payload, expected_version=card["version"]
        )
    else:
        from tldw_chatbook.UI.CCP_Modules import ccp_character_handler

        monkeypatch.setattr(ccp_character_handler, "_default_character_db", lambda: db)
        assert ccp_character_handler.update_character(str(character), payload)
    return db.get_character_card_by_id(character)


def _spy_world_book_fetches(monkeypatch):
    fetches = []
    original = WorldBookManager.get_world_books_for_conversation

    def count(self, *args, **kwargs):
        fetches.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(WorldBookManager, "get_world_books_for_conversation", count)
    return fetches


@pytest.mark.parametrize("writer", ["persona", "ccp"])
def test_book_bearing_card_save_refetches_conversation_books(db, monkeypatch, writer):
    """TASK-34436 (a): the spy form — after a book-bearing card save the next
    world-info resolve re-FETCHES the conversation books (the cached processor
    was dropped), rather than merely re-rendering stale state."""
    conversation = db.add_conversation({"title": "Spy"})
    manager = WorldBookManager(db)
    book = manager.create_world_book("SpyBook")
    manager.create_world_book_entry(book, keys=["keep"], content="KEPT")
    manager.associate_world_book_with_conversation(conversation, book)
    character = db.add_character_card(
        {"name": "Spy", "extensions": _extensions("OLD")}
    )
    card = db.get_character_card_by_id(character)
    fetches = _spy_world_book_fetches(monkeypatch)

    assert "OLD" in resolver.apply_world_info_to_message(
        db, conversation, card, "castle", []
    )
    assert len(fetches) == 1
    assert "OLD" in resolver.apply_world_info_to_message(
        db, conversation, card, "castle", []
    )
    assert len(fetches) == 1  # warm: the cached processor skips the fetch

    updated = _save_card(
        db, monkeypatch, writer, character, card, {"extensions": _extensions("NEW")}
    )
    assert "NEW" in resolver.apply_world_info_to_message(
        db, conversation, updated, "castle", []
    )
    assert len(fetches) == 2  # invalidated -> the manager fetch ran again
    assert "NEW" in resolver.apply_world_info_to_message(
        db, conversation, updated, "castle", []
    )
    assert len(fetches) == 2  # re-warmed under the new card version


@pytest.mark.parametrize("writer", ["persona", "ccp"])
def test_dictionary_bearing_card_save_refetches_the_bundle(db, monkeypatch, writer):
    """TASK-34436 (b): the dictionary half — a card save that changes embedded
    dictionaries drops the warm bundle; the rebuild re-loads the conversation's
    attached dictionaries from the store."""
    conversation = db.add_conversation({"title": "DictSpy"})
    attached = dictionaries.save_chat_dictionary(
        db,
        "Attached",
        entries=[dictionaries.ChatDictionary(key="keep", content="KEPT")],
    )
    LocalChatDictionaryService(db).attach_to_conversation(attached, conversation)
    character = db.add_character_card(
        {"name": "DictSpy", "extensions": _extensions("OLD")}
    )
    card = db.get_character_card_by_id(character)

    loads = []
    original = dictionaries.load_chat_dictionary

    def count(*args, **kwargs):
        loads.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(dictionaries, "load_chat_dictionary", count)

    assert "OLD" in dictionaries.apply_active_chatdicts_to_text(
        db, conversation, card, "castle"
    )
    assert len(loads) == 1
    assert "OLD" in dictionaries.apply_active_chatdicts_to_text(
        db, conversation, card, "castle"
    )
    assert len(loads) == 1  # warm bundle: no dictionary loads

    updated = _save_card(
        db, monkeypatch, writer, character, card, {"extensions": _extensions("NEW")}
    )
    assert "NEW" in dictionaries.apply_active_chatdicts_to_text(
        db, conversation, updated, "castle"
    )
    assert len(loads) == 2  # invalidated -> the attached dictionary re-loaded
    assert "NEW" in dictionaries.apply_active_chatdicts_to_text(
        db, conversation, updated, "castle"
    )
    assert len(loads) == 2  # re-warmed under the new card version


@pytest.mark.parametrize("writer", ["persona", "ccp"])
def test_card_save_without_book_or_dictionary_fields_rebuilds_identically(
    db, inject, monkeypatch, writer
):
    """TASK-34436 (c): the invalidation is unconditional by design — ANY card
    save changes the version token, so a save that never touches book or
    dictionary fields still rebuilds the cache (refetches) with byte-identical
    output. Safe direction: extra work only, never stale content."""
    conversation = db.add_conversation({"title": "Safe"})
    character = db.add_character_card(
        {"name": "Safe", "extensions": _extensions("KEEP")}
    )
    card = db.get_character_card_by_id(character)

    if inject.kind == "world-info":
        rebuilds = _spy_world_book_fetches(monkeypatch)
    else:
        rebuilds = []
        original = dictionaries.load_character_dictionaries

        def count(*args, **kwargs):
            rebuilds.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(dictionaries, "load_character_dictionaries", count)

    before = inject(db, conversation, card)
    assert "KEEP" in before
    assert inject(db, conversation, card) == before  # warm
    assert len(rebuilds) == 1

    updated = _save_card(
        db, monkeypatch, writer, character, card, {"description": "unrelated edit"}
    )
    assert updated["version"] == card["version"] + 1
    assert inject(db, conversation, updated) == before  # identical output...
    assert len(rebuilds) == 2  # ...but the cache still rebuilt (safe direction)


def test_card_save_invalidation_matches_store_write_invalidation(
    db, monkeypatch
):
    """TASK-34436 (d): a book-bearing card save drives the cache through the
    same state transition as a direct WorldBookManager write — exactly one
    refetch on the next resolve, then warm again."""
    conversation = db.add_conversation({"title": "Equiv"})
    manager = WorldBookManager(db)
    book = manager.create_world_book("EquivBook")
    entry = manager.create_world_book_entry(book, keys=["castle"], content="OLD")
    manager.associate_world_book_with_conversation(conversation, book)
    character = db.add_character_card(
        {"name": "Equiv", "extensions": _extensions("CARD-OLD")}
    )
    card = db.get_character_card_by_id(character)
    fetches = _spy_world_book_fetches(monkeypatch)

    def text(active_card):
        return resolver.apply_world_info_to_message(
            db, conversation, active_card, "castle", []
        )

    assert "CARD-OLD" in text(card)
    assert len(fetches) == 1
    assert "CARD-OLD" in text(card)
    assert len(fetches) == 1

    # Store-write invalidation (the ADR-221 baseline behavior).
    assert manager.update_world_book_entry(entry, content="NEW")
    assert "NEW" in text(card)
    assert len(fetches) == 2  # refetched exactly once...
    assert "NEW" in text(card)
    assert len(fetches) == 2  # ...then warm again

    # Card-save invalidation: the same transition, same observables.
    updated = _save_card(
        db,
        monkeypatch,
        "persona",
        character,
        card,
        {"extensions": _extensions("CARD-NEW")},
    )
    assert "CARD-NEW" in text(updated)
    assert len(fetches) == 3  # refetched exactly once...
    assert "CARD-NEW" in text(updated)
    assert len(fetches) == 3  # ...then warm again


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
