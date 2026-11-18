"""Original scope presentation and selected scope refresh ownership."""

import asyncio
import sys

import pytest

from Tests.private_profile import private_profile_test


@pytest.mark.bootstrap_profile
def test_scope_projection_keeps_disabled_shadowed_and_attached_facts():
    from tldw_chatbook.Chat.console_execution_inspector_values import build_inspector_scopes, inspector_scopes

    value = build_inspector_scopes(
        "conversation",
        {"dictionaries": [
            {"name": "Local", "source": "conversation", "enabled": False, "shadowed": True},
            {"name": "Embedded", "source": "character"},
        ]},
        {"world_books": [{"name": "World", "entry_count": 3, "enabled": False}]},
    )
    assert [row.value for row in value.dictionary_rows] == [
        "from conversation (shadowed) (disabled)", "from character",
    ]
    assert value.world_book_rows[0].value == "3 entries (disabled)"
    assert all(action.enabled for action in (*value.dictionary_actions, *value.world_book_actions))
    encoded = inspector_scopes(value, False)
    assert inspector_scopes(encoded, True) == value
    encoded["dictionary_actions"][0]["enabled"] = 1
    with pytest.raises(ValueError):
        inspector_scopes(encoded, True)


@pytest.mark.bootstrap_profile
def test_original_no_scope_and_queried_empty_scope_remain_distinct():
    from tldw_chatbook.Chat.console_execution_inspector_values import build_inspector_scopes

    absent = build_inspector_scopes(None, {"dictionaries": []}, {"world_books": []})
    empty = build_inspector_scopes("conversation", {"dictionaries": []}, {"world_books": []})
    assert absent.dictionary_rows[0].label == absent.world_book_rows[0].label == "No active chat"
    assert not any(action.enabled for action in (*absent.dictionary_actions, *absent.world_book_actions))
    assert empty.dictionary_rows[0].label == "No dictionaries in play"
    assert empty.world_book_rows[0].label == "No world books in play"
    assert empty.dictionary_actions[0].enabled and not empty.dictionary_actions[1].enabled
    assert empty.world_book_actions[0].enabled and not empty.world_book_actions[1].enabled


@private_profile_test
@pytest.mark.asyncio
async def test_backend_scope_summaries_reuse_actual_reads_and_preserve_explicit_refresh(request):
    from Tests.Chat.test_console_control_trace import _observe
    from Tests.Chat.test_console_execution_profile import _settled_profile
    from tldw_chatbook.Backup_Recovery.chat_source_participants import build_dictionary_service
    from tldw_chatbook.Character_Chat import Chat_Dictionary_Lib, world_info_resolver
    from tldw_chatbook.Character_Chat.world_book_manager import WorldBookManager
    from tldw_chatbook.Chat.console_execution_backend import ConsoleExecutionBackend
    from tldw_chatbook.Chat.console_execution_profile import open_console_backend_profile
    from tldw_chatbook.Chat.console_execution_protocol import ConsoleWireMessage, WireIdentity
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
    from tldw_chatbook.DB.base_db import operation_owned_connection

    path, directory, *_ = _settled_profile()
    profile = open_console_backend_profile(expected_config_path=path, expected_data_directory=directory)
    backend = ConsoleExecutionBackend(profile, profile_epoch="profile", service_epoch="inspector-scopes")
    dictionary_reads, world_reads = [], []
    try:
        backend.start()
        session = backend.store.create_session(
            title="Inspector scope", assistant_kind="generic",
            settings=ConsoleSessionSettings(provider="openai", model=None, source="user"),
        )

        async def projection(refresh=None):
            sequence = backend._sequence + 1
            payload = {} if refresh is None else {"known_display_revision": None, "inspector_refresh": refresh}
            receipt = backend.dispatch(ConsoleWireMessage(
                WireIdentity("profile", "inspector-scopes", sequence, f"scope-{sequence}",
                             session.id, session.incarnation_id, backend.controls.source_revision),
                "command", "ReadProjection", payload,
            ))
            try:
                return await asyncio.shield(receipt.applied)
            finally:
                assert await asyncio.shield(receipt.retired) is True

        with (
            _observe(Chat_Dictionary_Lib.summarize_active_dictionaries.__code__, sys.monitoring.events.PY_START,
                     lambda _frame: dictionary_reads.append(True)),
            _observe(world_info_resolver.summarize_active_world_books.__code__, sys.monitoring.events.PY_START,
                     lambda _frame: world_reads.append(True)),
        ):
            await projection()
            owner = backend.presentation.inspector_scopes
            assert not dictionary_reads and not world_reads and owner.service is None
            database = profile.storage.chachanotes
            with operation_owned_connection(database):
                conversation_id = database.add_conversation({"id": "inspector-scope-conversation", "title": "Scope"})
                assert conversation_id
                service = build_dictionary_service(database)
                dictionary = service.create_dictionary({"name": "Scope dictionary"})
                service.attach_to_conversation(dictionary["id"], conversation_id)
                books = WorldBookManager(database)
                book_id = books.create_world_book("Scope world book")
                books.create_world_book_entry(book_id, keys=["scope"], content="Actual entry")
                books.associate_world_book_with_conversation(conversation_id, book_id)
            backend.store.publish_first_persisted_conversation(session.id, conversation_id)
            await projection()
            captured = owner.value
            assert len(dictionary_reads) == len(world_reads) == 1
            assert any(row.label == "Scope dictionary" for row in captured.dictionary_rows)
            assert any(row.label == "Scope world book" and row.value == "1 entries" for row in captured.world_book_rows)
            assert not owner.uncertain and not owner.reads

            backend.store.set_session_one_shot_prefill(session.id, "Actual display change")
            await projection()
            assert owner.value is captured
            assert len(dictionary_reads) == len(world_reads) == 1

            await projection("dictionary")
            assert len(dictionary_reads) == 2 and len(world_reads) == 1
            await projection("world_book")
            assert len(dictionary_reads) == len(world_reads) == 2
            assert not owner.uncertain and not owner.reads
    finally:
        await backend.aclose()
