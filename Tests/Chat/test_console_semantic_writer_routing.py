"""Route-level contracts for semantic message persistence."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import inspect

import pytest

from tldw_chatbook.Character_Chat.Character_Chat_Lib import (
    add_message_to_conversation,
)
from tldw_chatbook.Character_Chat.local_character_persona_service import (
    LocalCharacterPersonaService,
)
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, ConflictError, InputError
from tldw_chatbook.Research_Interop.chat_handoff import (
    insert_research_completion_message,
)
from tldw_chatbook.Sync_Interop.hashing import canonical_payload_hash


@pytest.fixture
def db() -> CharactersRAGDB:
    database = CharactersRAGDB(":memory:", "semantic-writer-routing")
    yield database
    database.close_connection()


def _conversation(db: CharactersRAGDB) -> str:
    conversation_id = db.add_conversation({"title": "semantic writer routing"})
    assert conversation_id is not None
    return conversation_id


def _revision_rows(db: CharactersRAGDB, message_id: str) -> list[sqlite3.Row]:
    return list(
        db.get_connection().execute(
            """
            SELECT revision_id, revision_sequence, predecessor_revision_id,
                   live_message_id, live_locator_retired_at
              FROM console_trace_semantic_revisions
             WHERE source_message_id = ?
             ORDER BY revision_sequence
            """,
            (message_id,),
        )
    )


def _epoch(db: CharactersRAGDB) -> int:
    return int(
        db.get_connection()
        .execute("SELECT epoch FROM console_trace_graph_epoch WHERE singleton_id = 1")
        .fetchone()[0]
    )


def _semantic_bytes(db: CharactersRAGDB, message_id: str) -> tuple[object, ...]:
    row = (
        db.get_connection()
        .execute(
            """
        SELECT sender, role, content, image_data, image_mime_type,
               provider_continuation_json, thinking_blocks_json,
               assistant_generation_state
          FROM messages WHERE id = ?
        """,
            (message_id,),
        )
        .fetchone()
    )
    assert row is not None
    return tuple(row)


def test_create_message_records_complete_initial_envelope(db: CharactersRAGDB) -> None:
    service = ChatPersistenceService(db)
    conversation_id = _conversation(db)

    message_id = service.create_message(
        conversation_id=conversation_id,
        sender="assistant",
        content="created",
        image_data=b"primary",
        image_mime_type="image/png",
        message_id="created-message",
        attachments=(
            {
                "position": 0,
                "data": b"primary",
                "mime_type": "image/png",
                "display_name": "primary",
            },
            {
                "position": 1,
                "data": b"variant",
                "mime_type": "image/webp",
                "display_name": "variant",
            },
        ),
    )
    revisions = _revision_rows(db, message_id)
    assert [row["revision_sequence"] for row in revisions] == [0]
    assert revisions[0]["live_message_id"] == message_id
    assert (
        db.get_connection()
        .execute(
            "SELECT data FROM message_attachments WHERE message_id = ? AND position = 1",
            (message_id,),
        )
        .fetchone()[0]
        == b"variant"
    )


def test_create_message_feedback_is_part_of_one_initial_commit(
    db: CharactersRAGDB,
) -> None:
    service = ChatPersistenceService(db)
    conversation_id = _conversation(db)
    before_epoch = _epoch(db)

    message_id = service.create_message(
        conversation_id=conversation_id,
        sender="assistant",
        content="created",
        message_id="created-with-feedback",
        feedback="1;reviewed",
    )

    assert [row["revision_sequence"] for row in _revision_rows(db, message_id)] == [0]
    assert db.get_message_by_id_without_blob(message_id)["feedback"] == "1;reviewed"
    assert _epoch(db) == before_epoch + 1


def test_add_message_exposes_no_live_cursor_callback_and_composite_revalidates(
    db: CharactersRAGDB,
) -> None:
    assert (
        "_before_initial_semantic_revision"
        not in inspect.signature(db.add_message).parameters
    )
    conversation_id = _conversation(db)
    before_epoch = _epoch(db)

    with pytest.raises(InputError):
        db.add_message_with_semantic_sidecars(
            {
                "id": "invalid-composite",
                "conversation_id": conversation_id,
                "sender": "user",
                "role": "user",
                "content": "",
                "provider_continuation_json": json.dumps(
                    {
                        "schema_version": 1,
                        "checkpoint_revision": 1,
                        "provider": "deepseek",
                        "protocol": "responses",
                        "model": "deepseek-test",
                        "api_base_url": "https://api.deepseek.com/v1",
                        "state": "active",
                        "rounds": [],
                    }
                ),
            },
            attachments=(),
            generation_metadata=None,
            feedback=None,
        )

    assert db.get_message_by_id("invalid-composite") is None
    assert _epoch(db) == before_epoch


@pytest.mark.parametrize(
    "rows", ([], [{"position": 1, "data": b"x", "mime_type": "image/png"}])
)
def test_missing_attachment_owner_raises_conflict_for_empty_and_nonempty_rows(
    db: CharactersRAGDB, rows: list[dict[str, object]]
) -> None:
    with pytest.raises(ConflictError):
        db.set_message_attachments("missing-message", rows)


def test_classic_persona_and_research_owners_roll_back_before_initial_revision(
    db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    conversation_id = db.add_conversation(
        {
            "title": "owner routes",
            "assistant_kind": "persona",
            "assistant_id": "persona-1",
            "discovery_owner": "ccp_persona",
            "discovery_entity_id": "persona-1",
        }
    )
    assert conversation_id is not None
    persona = LocalCharacterPersonaService(db)
    original = db._ensure_initial_semantic_revision

    def fail_revision(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected initial revision failure")

    monkeypatch.setattr(db, "_ensure_initial_semantic_revision", fail_revision)

    assert (
        add_message_to_conversation(
            db, conversation_id, sender="user", content="classic"
        )
        is None
    )
    with pytest.raises(RuntimeError, match="initial revision failure"):
        persona.create_character_chat_message(
            conversation_id, {"role": "user", "content": "persona"}
        )
    assert (
        insert_research_completion_message(
            db,
            {
                "run_id": "run-1",
                "question": "question",
                "report_markdown": "research",
                "chat_handoff": {"conversation_id": conversation_id},
            },
        )
        is None
    )
    assert db.get_messages_for_conversation(conversation_id) == []

    monkeypatch.setattr(db, "_ensure_initial_semantic_revision", original)
    created_ids = [
        add_message_to_conversation(
            db, conversation_id, sender="user", content="classic"
        ),
        persona.create_character_chat_message(
            conversation_id, {"role": "user", "content": "persona"}
        )["id"],
        insert_research_completion_message(
            db,
            {
                "run_id": "run-2",
                "question": "question",
                "report_markdown": "research",
                "chat_handoff": {"conversation_id": conversation_id},
            },
        ),
    ]
    assert all(created_ids)
    assert all(
        [row["revision_sequence"] for row in _revision_rows(db, str(message_id))] == [0]
        for message_id in created_ids
    )


@pytest.mark.parametrize(
    ("route", "expected_error"),
    [
        ("update-empty", InputError),
        ("update-missing", ConflictError),
        ("continuation-invalid", InputError),
        ("continuation-missing", ConflictError),
        ("generation-invalid", InputError),
        ("generation-missing", ConflictError),
    ],
)
def test_semantic_public_routes_validate_before_coordinator_envelope_read(
    db: CharactersRAGDB,
    route: str,
    expected_error: type[Exception],
) -> None:
    kwargs = {
        "message_id": "missing-message",
        "expected_message_version": 1,
        "provider_continuation_json": None,
    }

    with pytest.raises(expected_error):
        if route == "update-empty":
            db.update_message("missing-message", {}, 1)
        elif route == "update-missing":
            db.update_message("missing-message", {"content": "after"}, 1)
        elif route == "continuation-invalid":
            db.update_provider_continuation(
                **dict(kwargs, message_id="", expected_message_version=0)
            )
        elif route == "continuation-missing":
            db.update_provider_continuation(**kwargs)
        elif route == "generation-invalid":
            db.replace_assistant_generation_projection(
                message_id="",
                content="answer",
                thinking_blocks_json=None,
                provider_continuation_json=None,
                assistant_generation_state="complete",
                usage_json=None,
            )
        else:
            db.replace_assistant_generation_projection(
                message_id="missing-message",
                content="answer",
                thinking_blocks_json=None,
                provider_continuation_json=None,
                assistant_generation_state="complete",
                usage_json=None,
            )


@pytest.mark.parametrize(
    "route",
    (
        "update",
        "continuation",
        "generation",
        "attachment-empty",
        "attachment-nonempty",
    ),
)
def test_soft_deleted_public_routes_reject_before_coordinator_envelope_read(
    db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    conversation_id = _conversation(db)
    message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "assistant",
            "role": "assistant",
            "content": "deleted",
        }
    )
    assert message_id is not None
    assert db.soft_delete_message(message_id, expected_version=1)

    def coordinator_must_not_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("coordinator envelope read ran before public validation")

    monkeypatch.setattr(db, "_coordinate_semantic_mutation", coordinator_must_not_run)
    with pytest.raises(ConflictError):
        if route == "update":
            db.update_message(message_id, {"content": "after"}, expected_version=2)
        elif route == "continuation":
            db.update_provider_continuation(
                message_id=message_id,
                expected_message_version=2,
                provider_continuation_json=None,
            )
        elif route == "generation":
            db.replace_assistant_generation_projection(
                message_id=message_id,
                content="after",
                thinking_blocks_json=None,
                provider_continuation_json=None,
                assistant_generation_state="complete",
                usage_json=None,
                expected_version=2,
            )
        elif route == "attachment-empty":
            db.set_message_attachments(message_id, [])
        else:
            db.set_message_attachments(
                message_id,
                [{"position": 1, "data": b"x", "mime_type": "image/png"}],
            )


def test_update_message_content_creates_one_successor(db: CharactersRAGDB) -> None:
    service = ChatPersistenceService(db)
    conversation_id = _conversation(db)
    message_id = service.create_message(
        conversation_id=conversation_id,
        sender="user",
        content="before",
        message_id="edited-message",
    )
    before_epoch = _epoch(db)

    assert service.update_message_content(
        message_id=message_id,
        content="after",
        image_data=None,
        image_mime_type=None,
    )

    revisions = _revision_rows(db, message_id)
    assert [row["revision_sequence"] for row in revisions] == [0, 1]
    assert revisions[1]["predecessor_revision_id"] == revisions[0]["revision_id"]
    assert revisions[0]["live_message_id"] is None
    assert revisions[1]["live_message_id"] == message_id
    assert _epoch(db) == before_epoch + 1


def test_failed_attachment_rewrite_rolls_back_message_and_lineage(
    db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = ChatPersistenceService(db)
    conversation_id = _conversation(db)
    message_id = service.create_message(
        conversation_id=conversation_id,
        sender="user",
        content="before",
        message_id="rollback-message",
    )
    before_row = _semantic_bytes(db, message_id)
    before_revisions = [tuple(row) for row in _revision_rows(db, message_id)]
    before_epoch = _epoch(db)

    def fail_attachments(
        _cursor: sqlite3.Cursor, _message_id: str, _rows: list[dict]
    ) -> None:
        raise sqlite3.OperationalError("injected attachment failure")

    monkeypatch.setattr(db, "_set_message_attachments_uncoordinated", fail_attachments)
    with pytest.raises(sqlite3.OperationalError, match="injected attachment failure"):
        service.update_message_content(
            message_id=message_id,
            content="must roll back",
            image_data=None,
            image_mime_type=None,
            attachments=(),
        )

    assert _semantic_bytes(db, message_id) == before_row
    assert [tuple(row) for row in _revision_rows(db, message_id)] == before_revisions
    assert _epoch(db) == before_epoch


def test_content_and_attachment_update_has_one_successor_and_epoch(
    db: CharactersRAGDB,
) -> None:
    service = ChatPersistenceService(db)
    conversation_id = _conversation(db)
    message_id = service.create_message(
        conversation_id=conversation_id,
        sender="user",
        content="before",
        message_id="composite-update",
        attachments=(
            {
                "position": 0,
                "data": b"old-primary",
                "mime_type": "image/png",
            },
            {
                "position": 1,
                "data": b"old-extra",
                "mime_type": "image/png",
            },
        ),
    )
    before_epoch = _epoch(db)

    assert service.update_message_content(
        message_id=message_id,
        content="after",
        image_data=None,
        image_mime_type=None,
        attachments=(
            {
                "position": 0,
                "data": b"new-primary",
                "mime_type": "image/webp",
            },
            {
                "position": 1,
                "data": b"new-extra",
                "mime_type": "image/webp",
            },
        ),
    )

    revisions = _revision_rows(db, message_id)
    assert [row["revision_sequence"] for row in revisions] == [0, 1]
    assert _epoch(db) == before_epoch + 1
    assert (
        db.get_connection()
        .execute(
            "SELECT data FROM message_attachments WHERE message_id = ? AND position = 1",
            (message_id,),
        )
        .fetchone()[0]
        == b"new-extra"
    )


def test_attachment_append_and_selection_each_create_successor(
    db: CharactersRAGDB,
) -> None:
    service = ChatPersistenceService(db)
    conversation_id = _conversation(db)
    message_id = service.create_message(
        conversation_id=conversation_id,
        sender="assistant",
        content="image",
        image_data=b"primary",
        image_mime_type="image/png",
        message_id="attachment-message",
    )

    position = service.append_message_attachment(
        message_id,
        data=b"variant",
        mime_type="image/webp",
    )
    service.keep_message_attachment(message_id, position)

    revisions = _revision_rows(db, message_id)
    assert [row["revision_sequence"] for row in revisions] == [0, 1, 2]
    assert db.get_message_by_id(message_id)["image_data"] == b"variant"


def test_generation_and_continuation_replacements_each_create_successor(
    db: CharactersRAGDB,
) -> None:
    service = ChatPersistenceService(db)
    conversation_id = _conversation(db)
    message_id = service.create_message(
        conversation_id=conversation_id,
        sender="assistant",
        content="first generation",
        message_id="generation-message",
    )

    committed_version = service.replace_assistant_generation_projection(
        message_id=message_id,
        content="second generation",
        thinking_blocks_json=None,
        provider_continuation_json=None,
        assistant_generation_state="complete",
        usage_json=None,
        expected_version=1,
    )
    assert db.update_provider_continuation(
        message_id=message_id,
        expected_message_version=committed_version,
        provider_continuation_json=None,
        content="continuation replaced",
        assistant_generation_state="stopped",
    )

    revisions = _revision_rows(db, message_id)
    assert [row["revision_sequence"] for row in revisions] == [0, 1, 2]
    assert revisions[2]["predecessor_revision_id"] == revisions[1]["revision_id"]


def test_variant_creation_tracks_new_bytes_and_selection_only_fences_visibility(
    db: CharactersRAGDB,
) -> None:
    conversation_id = _conversation(db)
    original_id = db.add_message(
        {
            "id": "original-variant",
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "original",
        }
    )
    assert original_id is not None
    variant_id = db.create_message_variant(
        original_id,
        "alternative",
        is_selected=True,
    )
    assert variant_id is not None
    original_lineage = [tuple(row) for row in _revision_rows(db, original_id)]
    variant_lineage = [tuple(row) for row in _revision_rows(db, variant_id)]
    before_epoch = _epoch(db)

    assert db.select_message_variant(original_id)

    assert [tuple(row) for row in _revision_rows(db, original_id)] == original_lineage
    assert [tuple(row) for row in _revision_rows(db, variant_id)] == variant_lineage
    assert _epoch(db) == before_epoch + 1


def test_soft_delete_preserves_semantic_bytes_and_revision_lineage(
    db: CharactersRAGDB,
) -> None:
    conversation_id = _conversation(db)
    message_id = db.create_assistant_with_continuation(
        message_id="soft-deleted-message",
        conversation_id=conversation_id,
        parent_message_id=None,
        content="answer",
        provider_continuation_json=json.dumps(
            {
                "schema_version": 1,
                "checkpoint_revision": 1,
                "provider": "deepseek",
                "protocol": "responses",
                "model": "deepseek-test",
                "api_base_url": "https://api.deepseek.com/v1",
                "state": "active",
                "rounds": [
                    {
                        "assistant_content": "answer",
                        "reasoning_blocks": [],
                        "calls": [
                            {
                                "call_id": "call-1",
                                "name": "calculator",
                                "arguments": "{}",
                                "state": "pending",
                            }
                        ],
                    }
                ],
            }
        ),
    )
    before_bytes = _semantic_bytes(db, message_id)
    before_revisions = [tuple(row) for row in _revision_rows(db, message_id)]
    before_epoch = _epoch(db)

    assert db.soft_delete_message(message_id, expected_version=1)

    assert _semantic_bytes(db, message_id) == before_bytes
    assert [tuple(row) for row in _revision_rows(db, message_id)] == before_revisions
    assert _epoch(db) == before_epoch + 1
    inventory = (
        Path(__file__).resolve().parents[2]
        / "Docs/Development/console-semantic-mutation-inventory.md"
    ).read_text(encoding="utf-8")
    for route in ("soft_delete_message", "soft_delete_message_subtree"):
        assert (
            f"CharactersRAGDB.{route}::sql:update:messages` — visibility/ownership-only"
            in inventory
        )


def test_subtree_soft_delete_retains_every_envelope_and_rolls_back_exactly(
    db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    conversation_id = _conversation(db)
    message_ids: list[str] = []
    parent_id: str | None = None
    for index in range(3):
        message_id = db.add_message(
            {
                "id": f"subtree-{index}",
                "conversation_id": conversation_id,
                "parent_message_id": parent_id,
                "sender": "assistant" if index % 2 else "user",
                "content": f"retained-{index}",
            }
        )
        assert message_id is not None
        message_ids.append(message_id)
        db.set_message_attachments(
            message_id,
            (
                {
                    "position": 1,
                    "data": f"attachment-{index}-1".encode(),
                    "mime_type": "application/octet-stream",
                    "display_name": f"attachment-{index}-1",
                },
                {
                    "position": 2,
                    "data": f"attachment-{index}-2".encode(),
                    "mime_type": "application/octet-stream",
                    "display_name": f"attachment-{index}-2",
                },
            ),
        )
        parent_id = message_id

    before_bytes = {
        message_id: _semantic_bytes(db, message_id) for message_id in message_ids
    }
    before_lineage = {
        message_id: [tuple(row) for row in _revision_rows(db, message_id)]
        for message_id in message_ids
    }
    before_rows = {
        row["id"]: (row["deleted"], row["version"])
        for row in db.get_connection()
        .execute(
            "SELECT id, deleted, version FROM messages WHERE id IN (?, ?, ?)",
            tuple(message_ids),
        )
        .fetchall()
    }
    before_epoch = _epoch(db)
    before_attachments = [
        tuple(row)
        for row in db.get_connection()
        .execute(
            "SELECT message_id, position, data, mime_type, display_name "
            "FROM message_attachments WHERE message_id IN (?, ?, ?) "
            "ORDER BY message_id, position",
            tuple(message_ids),
        )
        .fetchall()
    ]
    original_advance = db._advance_semantic_graph_epoch

    def fail_epoch(_cursor: sqlite3.Cursor) -> None:
        raise RuntimeError("injected subtree checkpoint failure")

    monkeypatch.setattr(db, "_advance_semantic_graph_epoch", fail_epoch)
    with pytest.raises(RuntimeError, match="subtree checkpoint failure"):
        db.soft_delete_message_subtree(message_ids[0], expected_version=1)

    rolled_back = {
        row["id"]: (row["deleted"], row["version"])
        for row in db.get_connection()
        .execute(
            "SELECT id, deleted, version FROM messages WHERE id IN (?, ?, ?)",
            tuple(message_ids),
        )
        .fetchall()
    }
    assert rolled_back == before_rows
    assert _epoch(db) == before_epoch
    assert [
        tuple(row)
        for row in db.get_connection()
        .execute(
            "SELECT message_id, position, data, mime_type, display_name "
            "FROM message_attachments WHERE message_id IN (?, ?, ?) "
            "ORDER BY message_id, position",
            tuple(message_ids),
        )
        .fetchall()
    ] == before_attachments
    for message_id in message_ids:
        assert _semantic_bytes(db, message_id) == before_bytes[message_id]
        assert [tuple(row) for row in _revision_rows(db, message_id)] == before_lineage[
            message_id
        ]

    monkeypatch.setattr(db, "_advance_semantic_graph_epoch", original_advance)
    tombstones = db.soft_delete_message_subtree(message_ids[0], expected_version=1)
    assert {row["message_id"] for row in tombstones} == set(message_ids)
    assert _epoch(db) == before_epoch + 1
    assert [
        tuple(row)
        for row in db.get_connection()
        .execute(
            "SELECT message_id, position, data, mime_type, display_name "
            "FROM message_attachments WHERE message_id IN (?, ?, ?) "
            "ORDER BY message_id, position",
            tuple(message_ids),
        )
        .fetchall()
    ] == before_attachments
    for message_id in message_ids:
        row = (
            db.get_connection()
            .execute(
                "SELECT deleted, version FROM messages WHERE id = ?", (message_id,)
            )
            .fetchone()
        )
        assert (row["deleted"], row["version"]) == (1, 2)
        assert _semantic_bytes(db, message_id) == before_bytes[message_id]
        assert [
            tuple(item) for item in _revision_rows(db, message_id)
        ] == before_lineage[message_id]


def test_presentation_only_metadata_creates_no_revision_or_epoch(
    db: CharactersRAGDB,
) -> None:
    service = ChatPersistenceService(db)
    conversation_id = _conversation(db)
    message_id = service.create_message(
        conversation_id=conversation_id,
        sender="assistant",
        content="answer",
        message_id="metadata-message",
    )
    before_revisions = [tuple(row) for row in _revision_rows(db, message_id)]
    before_epoch = _epoch(db)

    assert service.update_message_usage(message_id=message_id, usage_json="{}")
    assert service.update_message_metadata(message_id=message_id, metadata_json="{}")

    assert [tuple(row) for row in _revision_rows(db, message_id)] == before_revisions
    assert _epoch(db) == before_epoch


def test_sync_update_creates_successor_and_tombstone_preserves_it(
    db: CharactersRAGDB,
) -> None:
    conversation_id = _conversation(db)
    stable_key = f"{conversation_id}:sync-message"
    initial = {
        "assistant_generation_state": None,
        "content": "before",
        "role": "user",
    }
    db.append_chat_message(stable_key, initial, canonical_payload_hash(initial))
    updated = dict(initial, content="after")

    db.append_chat_message(stable_key, updated, canonical_payload_hash(updated))
    before_delete = [tuple(row) for row in _revision_rows(db, "sync-message")]
    before_bytes = _semantic_bytes(db, "sync-message")

    db.delete_chat_message(
        stable_key,
        canonical_payload_hash({"deleted": True}),
    )

    assert len(before_delete) == 2
    assert [tuple(row) for row in _revision_rows(db, "sync-message")] == before_delete
    assert _semantic_bytes(db, "sync-message") == before_bytes
