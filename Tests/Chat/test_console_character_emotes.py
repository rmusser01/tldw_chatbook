"""Console store character-emote sanitization and terminal contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tldw_chatbook.Character_Chat.emote_directives import (
    CharacterEmoteAssetReference,
    CharacterEmoteRunSnapshot,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.unit


def _snapshot() -> CharacterEmoteRunSnapshot:
    return CharacterEmoteRunSnapshot(
        actor_id=7,
        pack_id=11,
        pack_version_id=13,
        states=("happy", "sad", "smug"),
        assets=(
            CharacterEmoteAssetReference("happy", "happy", 17, 19),
            CharacterEmoteAssetReference("sad", "sad", 23, 29),
            CharacterEmoteAssetReference("smug", "custom:smug", 31, 37),
        ),
    )


def _armed_message(store: ConsoleChatStore):
    session = store.create_session(
        assistant_kind="character",
        assistant_id="7",
        character_id=7,
    )
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="Please tell me the result.",
    )
    assistant = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
    )
    store.begin_character_emote_capture(assistant.id, _snapshot())
    return session, assistant


def test_armed_stream_strips_controls_and_publishes_every_event() -> None:
    store = ConsoleChatStore()
    session, assistant = _armed_message(store)

    store.append_stream_chunk(assistant.id, "Emo")
    assert store.get_message(assistant.id).content == ""
    store.append_stream_chunk(
        assistant.id,
        "te: smug\nHello\nEmote: sad\nDone",
    )

    assert store.get_message(assistant.id).content == "Hello\nDone"
    events = store.character_emote_events_after(session.id, 0)
    assert [(event.sequence, event.message_id, event.state) for event in events] == [
        (1, assistant.id, "smug"),
        (2, assistant.id, "sad"),
    ]

    completed = store.mark_message_complete(assistant.id)
    emote = completed.metadata.character_emote
    assert completed.content == "Hello\nDone"
    assert emote is not None
    assert emote.mood_label == "sad"
    assert [(event.state, event.at_char) for event in emote.emote_events] == [
        ("smug", 0),
        ("sad", 6),
    ]
    assert emote.actor_id == 7
    assert emote.pack_version_id == 13
    assert emote.expression_key == "sad"
    assert emote.asset_id == 23


def test_unarmed_generic_stream_remains_byte_for_byte_unchanged() -> None:
    store = ConsoleChatStore()
    session = store.create_session()
    assistant = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
    )

    store.append_stream_chunk(assistant.id, "Emote: sad\nVisible")
    completed = store.mark_message_complete(assistant.id)

    assert completed.content == "Emote: sad\nVisible"
    assert completed.metadata is None


def test_success_without_explicit_event_uses_pinned_heuristic() -> None:
    store = ConsoleChatStore()
    _session, assistant = _armed_message(store)

    store.append_stream_chunk(assistant.id, "I am happy and glad!")
    completed = store.mark_message_complete(assistant.id)

    emote = completed.metadata.character_emote
    assert emote is not None
    assert emote.emote_events == ()
    assert emote.mood_label == "happy"
    assert emote.mood_confidence is not None
    assert emote.expression_key == "happy"
    assert emote.asset_id == 17


@pytest.mark.parametrize(("terminal", "reason"), [("stopped", "stopped"), ("failed", "failed")])
def test_unsuccessful_terminal_discards_candidate_and_never_uses_heuristic(
    terminal: str,
    reason: str,
) -> None:
    store = ConsoleChatStore()
    _session, assistant = _armed_message(store)
    store.append_stream_chunk(assistant.id, "Visible\nEmote: happy")

    completed = getattr(store, f"mark_message_{terminal}")(assistant.id)

    assert completed.content == "Visible\n"
    emote = completed.metadata.character_emote
    assert emote is not None
    assert emote.mood_label is None
    assert emote.emote_events == ()
    assert emote.fallback_reason == reason


def test_citation_body_replacement_is_sanitized_and_replaces_durable_events() -> None:
    store = ConsoleChatStore()
    session = store.create_session(
        assistant_kind="character",
        assistant_id="7",
        character_id=7,
    )
    assistant = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        defer_terminal_persistence=True,
    )
    store.begin_character_emote_capture(assistant.id, _snapshot())
    store.append_stream_chunk(assistant.id, "Emote: happy\nProvisional")

    replaced = store.replace_deferred_terminal_body(
        assistant.id,
        "Emote: ../../bad\nSelected\nEmote: smug",
    )
    completed = store.mark_message_complete(assistant.id)

    assert replaced.content == "Selected\n"
    assert completed.content == "Selected\n"
    emote = completed.metadata.character_emote
    assert emote is not None
    assert [(event.state, event.at_char) for event in emote.emote_events] == [
        ("smug", 9)
    ]
    assert emote.mood_label == "smug"
    assert [event.state for event in store.character_emote_events_after(session.id, 0)] == [
        "happy",
        "smug",
    ]


def test_parser_fault_reprocesses_whole_chunk_with_eventless_fail_closed_sanitizer() -> None:
    store = ConsoleChatStore()
    _session, assistant = _armed_message(store)

    class FaultingParser:
        def safe_copy(self):
            from tldw_chatbook.Character_Chat.emote_directives import (
                CharacterEmoteStreamParser,
            )

            return CharacterEmoteStreamParser()

        def push(self, _chunk):
            raise RuntimeError("injected parser fault")

    store._character_emote_captures[assistant.id].parser = FaultingParser()

    store.append_stream_chunk(assistant.id, "Safe\nEmote: happy\nAfter")
    store.append_stream_chunk(assistant.id, "\nEmote: sad\nTail")
    completed = store.mark_message_complete(assistant.id)

    assert completed.content == "Safe\nAfter\nTail"
    emote = completed.metadata.character_emote
    assert emote is not None
    assert emote.emote_events == ()
    assert emote.fallback_reason == "parser_error"


def test_real_sqlite_persists_only_sanitized_text_and_bounded_metadata() -> None:
    db = CharactersRAGDB(":memory:", "emote-test")
    try:
        character_id = int(db.add_character_card({"name": "Emote test"}))
        store = ConsoleChatStore(persistence=ChatPersistenceService(db))
        session = store.create_session(
            assistant_kind="character",
            assistant_id=str(character_id),
            character_id=character_id,
        )
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content="Question",
            persist=True,
        )
        assistant = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="",
            persist=True,
        )
        store.begin_character_emote_capture(
            assistant.id,
            replace(_snapshot(), actor_id=character_id),
        )

        store.append_stream_chunk(
            assistant.id,
            "Emote: smug\nSafe answer\nEmote: ../../bad",
        )
        completed = store.mark_message_complete(assistant.id)
        row = db.get_message_by_id(completed.persisted_message_id)

        assert row["content"] == "Safe answer\n"
        assert "Emote:" not in row["content"]
        assert row["metadata_json"] is not None
        assert "Safe answer" not in row["metadata_json"]
        assert "smug" in row["metadata_json"]
    finally:
        db.close_connection()
