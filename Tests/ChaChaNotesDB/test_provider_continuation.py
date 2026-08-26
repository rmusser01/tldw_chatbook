"""Canonical provider continuation persistence on exact message owners."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tldw_chatbook.Chat.provider_continuation import (
    dump_provider_continuation_json,
    parse_provider_continuation_json,
)
from tldw_chatbook.DB.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)


def _checkpoint_json(*, canary: str = "private reasoning") -> str:
    return json.dumps(
        {
            "rounds": [
                {
                    "calls": [
                        {
                            "state": "pending",
                            "arguments": '{"expression":"2+2"}',
                            "name": "calculator",
                            "call_id": "call_1",
                        }
                    ],
                    "reasoning_blocks": [canary],
                    "assistant_content": "",
                }
            ],
            "state": "active",
            "api_base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-v4-flash",
            "protocol": "responses",
            "provider": "deepseek",
            "checkpoint_revision": 1,
            "schema_version": 1,
        },
        indent=2,
    )


def _kimi_checkpoint_json(
    content: str,
    *,
    post_tool: bool = False,
    canary: str = "private K3 reasoning",
    model: str = "kimi-k3",
) -> str:
    rounds: list[dict[str, object]] = []
    if post_tool:
        rounds.append(
            {
                "assistant_content": "",
                "reasoning_blocks": ["private tool reasoning"],
                "calls": [
                    {
                        "call_id": "call_1",
                        "name": "calculator",
                        "arguments": '{"expression":"2+2"}',
                        "state": "completed",
                        "result": "4",
                    }
                ],
            }
        )
    rounds.append(
        {
            "assistant_content": content,
            "reasoning_blocks": [canary],
            "calls": [],
        }
    )
    return json.dumps(
        {
            "schema_version": 1,
            "checkpoint_revision": 2 if post_tool else 1,
            "provider": "moonshot",
            "protocol": "chat_completions",
            "model": model,
            "api_base_url": "https://api.moonshot.ai/v1",
            "state": "complete",
            "rounds": rounds,
        }
    )


def _db_with_conversation(tmp_path: Path) -> tuple[CharactersRAGDB, str]:
    db = CharactersRAGDB(tmp_path / "continuation.db", client_id="continuation-test")
    conversation_id = db.add_conversation({"title": "continuation"})
    assert conversation_id is not None
    return db, conversation_id


def test_add_message_canonicalizes_and_round_trips_every_message_projection(
    tmp_path: Path,
) -> None:
    db, conversation_id = _db_with_conversation(tmp_path)
    raw = _checkpoint_json()
    canonical = dump_provider_continuation_json(parse_provider_continuation_json(raw))

    root_id = db.add_message(
        {
            "id": "assistant-owner",
            "conversation_id": conversation_id,
            "parent_message_id": None,
            "sender": "assistant",
            "role": "assistant",
            "content": "",
            "provider_continuation_json": raw,
        }
    )
    child_id = db.add_message(
        {
            "id": "assistant-child",
            "conversation_id": conversation_id,
            "parent_message_id": root_id,
            "sender": "assistant",
            "role": "assistant",
            "content": "visible",
            "provider_continuation_json": raw,
        }
    )

    assert root_id == "assistant-owner"
    assert db.get_message_by_id(root_id)["provider_continuation_json"] == canonical
    assert (
        db.get_latest_message_for_conversation(conversation_id)[
            "provider_continuation_json"
        ]
        == canonical
    )
    assert {
        row["id"]: row["provider_continuation_json"]
        for row in db.get_messages_for_conversation(conversation_id)
    } == {root_id: canonical, child_id: canonical}
    assert (
        db.get_root_messages_for_conversation(conversation_id, limit=10, offset=0)[0][
            "provider_continuation_json"
        ]
        == canonical
    )
    assert (
        db.get_messages_for_conversation_by_parent_ids(conversation_id, [root_id])[0][
            "provider_continuation_json"
        ]
        == canonical
    )
    assert db.search_messages_by_content("private reasoning") == []


def test_add_message_rejects_invalid_or_wrong_owner_without_private_error_context(
    tmp_path: Path,
) -> None:
    db, conversation_id = _db_with_conversation(tmp_path)
    canary = "PRIVATE-CANARY-DO-NOT-LOG"

    with pytest.raises(InputError) as invalid:
        db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "assistant",
                "role": "assistant",
                "content": "visible",
                "provider_continuation_json": canary,
            }
        )
    assert canary not in str(invalid.value)
    assert invalid.value.__cause__ is None
    assert invalid.value.__context__ is None

    with pytest.raises(InputError):
        db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "role": "user",
                "content": "",
                "provider_continuation_json": _checkpoint_json(),
            }
        )
    with pytest.raises(InputError):
        db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "assistant",
                "role": "assistant",
                "content": "",
            }
        )


def test_add_message_requires_exact_kimi_final_content_before_insert(
    tmp_path: Path,
) -> None:
    db, conversation_id = _db_with_conversation(tmp_path)
    canary = "PRIVATE-KIMI-ADD-CANARY"

    with pytest.raises(InputError) as invalid:
        db.add_message(
            {
                "id": "kimi-add-mismatch",
                "conversation_id": conversation_id,
                "sender": "assistant",
                "content": "visible answer",
                "provider_continuation_json": _kimi_checkpoint_json(
                    "different answer", canary=canary
                ),
            }
        )

    assert canary not in str(invalid.value)
    assert invalid.value.__cause__ is None
    assert invalid.value.__context__ is None
    assert db.get_message_by_id("kimi-add-mismatch") is None
    assert _message_sync_entries(db, "kimi-add-mismatch") == []

    checkpoint_json = _kimi_checkpoint_json("visible answer")
    message_id = db.add_message(
        {
            "id": "kimi-add-match",
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "visible answer",
            "provider_continuation_json": checkpoint_json,
        }
    )
    assert message_id == "kimi-add-match"
    assert db.get_message_by_id(message_id)[
        "provider_continuation_json"
    ] == dump_provider_continuation_json(
        parse_provider_continuation_json(checkpoint_json)
    )


def test_add_message_family_final_content_rule_covers_versioned_kimi(
    tmp_path: Path,
) -> None:
    """TASK-19170: the exact-final-content ownership rule follows the
    versioned-kimi family (k2.6 preserved-thinking checkpoints now exist),
    not the kimi-k3 literal."""
    db, conversation_id = _db_with_conversation(tmp_path)
    canary = "PRIVATE-KIMI-FAMILY-CANARY"

    checkpoint_json = _kimi_checkpoint_json("visible answer", model="kimi-k2.6")
    message_id = db.add_message(
        {
            "id": "kimi-family-add-match",
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "visible answer",
            "provider_continuation_json": checkpoint_json,
        }
    )
    assert message_id == "kimi-family-add-match"

    with pytest.raises(InputError) as invalid:
        db.add_message(
            {
                "id": "kimi-family-add-mismatch",
                "conversation_id": conversation_id,
                "sender": "assistant",
                "content": "visible answer",
                "provider_continuation_json": _kimi_checkpoint_json(
                    "different answer", model="kimi-k2.6", canary=canary
                ),
            }
        )

    assert canary not in str(invalid.value)
    assert db.get_message_by_id("kimi-family-add-mismatch") is None


def test_message_search_returns_public_fields_without_private_continuation(
    tmp_path: Path,
) -> None:
    db, conversation_id = _db_with_conversation(tmp_path)
    canary = "PRIVATE-SEARCH-CANARY"
    message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "public searchable answer",
            "provider_continuation_json": _checkpoint_json(canary=canary),
        }
    )

    hits = db.search_messages_by_content("searchable")

    assert len(hits) == 1
    assert hits[0]["id"] == message_id
    assert hits[0]["content"] == "public searchable answer"
    assert "provider_continuation_json" not in hits[0]
    assert canary not in json.dumps(hits[0], default=str)


def _message_sync_entries(
    db: CharactersRAGDB, message_id: str
) -> list[dict[str, object]]:
    rows = db.get_connection().execute(
        """
        SELECT operation, version, payload
          FROM sync_log
         WHERE entity = 'messages' AND entity_id = ?
         ORDER BY change_id
        """,
        (message_id,),
    )
    return [
        {
            "operation": row["operation"],
            "version": row["version"],
            "payload": json.loads(row["payload"]),
        }
        for row in rows
    ]


def test_atomic_create_preserves_id_version_and_exactly_one_sync_intent(
    tmp_path: Path,
) -> None:
    db, conversation_id = _db_with_conversation(tmp_path)
    conversation = db.get_conversation_by_id(conversation_id)
    assert conversation is not None
    message_id = "c15930c4-7596-4475-bdbf-742ed76d7c89"
    raw = _checkpoint_json()
    canonical = dump_provider_continuation_json(parse_provider_continuation_json(raw))

    created = db.create_assistant_with_continuation(
        message_id=message_id,
        conversation_id=conversation_id,
        parent_message_id=None,
        content="",
        provider_continuation_json=raw,
        expected_conversation_version=conversation["version"],
    )

    assert created == message_id
    row = db.get_message_by_id(message_id)
    assert row is not None
    assert row["version"] == 1
    assert row["provider_continuation_json"] == canonical
    entries = _message_sync_entries(db, message_id)
    assert len(entries) == 1
    assert entries[0]["operation"] == "create"
    assert entries[0]["version"] == 1
    assert entries[0]["payload"]["provider_continuation_json"] == canonical

    with pytest.raises(ConflictError):
        db.create_assistant_with_continuation(
            message_id=message_id,
            conversation_id=conversation_id,
            parent_message_id=None,
            content="",
            provider_continuation_json=raw,
        )
    assert len(_message_sync_entries(db, message_id)) == 1


def test_atomic_create_stale_invalid_and_crash_paths_leave_no_row_or_intent(
    tmp_path: Path,
) -> None:
    db, conversation_id = _db_with_conversation(tmp_path)
    conversation = db.get_conversation_by_id(conversation_id)
    assert conversation is not None
    raw = _checkpoint_json(canary="PRIVATE-CREATE-CANARY")

    with pytest.raises(ConflictError):
        db.create_assistant_with_continuation(
            message_id="stale-owner",
            conversation_id=conversation_id,
            parent_message_id=None,
            content="",
            provider_continuation_json=raw,
            expected_conversation_version=conversation["version"] + 1,
        )
    assert db.get_message_by_id("stale-owner") is None
    assert _message_sync_entries(db, "stale-owner") == []

    with pytest.raises(InputError) as invalid:
        db.create_assistant_with_continuation(
            message_id="invalid-owner",
            conversation_id=conversation_id,
            parent_message_id=None,
            content="",
            provider_continuation_json="PRIVATE-CREATE-CANARY",
        )
    assert "PRIVATE-CREATE-CANARY" not in str(invalid.value)
    assert invalid.value.__context__ is None
    assert db.get_message_by_id("invalid-owner") is None

    connection = db.get_connection()
    connection.execute(
        """
        CREATE TRIGGER inject_create_crash
        AFTER INSERT ON messages
        BEGIN
          SELECT RAISE(ABORT, 'injected crash');
        END
        """
    )
    connection.commit()
    with pytest.raises(CharactersRAGDBError):
        db.create_assistant_with_continuation(
            message_id="crash-owner",
            conversation_id=conversation_id,
            parent_message_id=None,
            content="",
            provider_continuation_json=raw,
        )
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM messages WHERE id = 'crash-owner'"
        ).fetchone()[0]
        == 0
    )
    assert _message_sync_entries(db, "crash-owner") == []


def test_atomic_create_requires_exact_kimi_final_content_without_partial_write(
    tmp_path: Path,
) -> None:
    db, conversation_id = _db_with_conversation(tmp_path)
    canary = "PRIVATE-KIMI-CREATE-CANARY"

    with pytest.raises(InputError) as invalid:
        db.create_assistant_with_continuation(
            message_id="kimi-mismatch",
            conversation_id=conversation_id,
            parent_message_id=None,
            content="visible answer",
            provider_continuation_json=_kimi_checkpoint_json(
                "different answer", canary=canary
            ),
        )

    assert canary not in str(invalid.value)
    assert invalid.value.__cause__ is None
    assert invalid.value.__context__ is None
    assert db.get_message_by_id("kimi-mismatch") is None
    assert _message_sync_entries(db, "kimi-mismatch") == []

    assert (
        db.create_assistant_with_continuation(
            message_id="kimi-equal",
            conversation_id=conversation_id,
            parent_message_id=None,
            content="visible answer",
            provider_continuation_json=_kimi_checkpoint_json("visible answer"),
        )
        == "kimi-equal"
    )


def test_atomic_update_is_optimistic_and_rolls_back_row_with_sync_intent(
    tmp_path: Path,
) -> None:
    db, conversation_id = _db_with_conversation(tmp_path)
    message_id = db.create_assistant_with_continuation(
        message_id="update-owner",
        conversation_id=conversation_id,
        parent_message_id=None,
        content="visible",
        provider_continuation_json=_checkpoint_json(),
    )
    before = db.get_message_by_id(message_id)
    assert before is not None

    assert db.update_provider_continuation(
        message_id=message_id,
        expected_message_version=before["version"],
        provider_continuation_json=_checkpoint_json(canary="next private"),
        content="updated visible",
    )
    after = db.get_message_by_id(message_id)
    assert after is not None
    assert after["content"] == "updated visible"
    assert after["version"] == before["version"] + 1
    entries = _message_sync_entries(db, message_id)
    assert len(entries) == 2
    assert entries[1]["operation"] == "update"
    assert entries[1]["version"] == after["version"]
    assert entries[1]["payload"]["content"] == "updated visible"
    assert (
        entries[1]["payload"]["provider_continuation_json"]
        == after["provider_continuation_json"]
    )

    stable = dict(after)
    with pytest.raises(ConflictError):
        db.update_provider_continuation(
            message_id=message_id,
            expected_message_version=before["version"],
            provider_continuation_json=None,
        )
    assert db.get_message_by_id(message_id) == stable
    assert len(_message_sync_entries(db, message_id)) == 2

    connection = db.get_connection()
    connection.execute(
        """
        CREATE TRIGGER inject_update_crash
        AFTER UPDATE ON messages
        WHEN NEW.id = 'update-owner'
        BEGIN
          SELECT RAISE(ABORT, 'injected crash');
        END
        """
    )
    connection.commit()
    with pytest.raises(CharactersRAGDBError):
        db.update_provider_continuation(
            message_id=message_id,
            expected_message_version=after["version"],
            provider_continuation_json=None,
        )
    assert db.get_message_by_id(message_id) == stable
    assert len(_message_sync_entries(db, message_id)) == 2


def test_atomic_update_requires_effective_kimi_final_content_and_rolls_back(
    tmp_path: Path,
) -> None:
    db, conversation_id = _db_with_conversation(tmp_path)
    message_id = db.create_assistant_with_continuation(
        message_id="kimi-update-owner",
        conversation_id=conversation_id,
        parent_message_id=None,
        content="current visible answer",
        provider_continuation_json=_checkpoint_json(),
    )
    before = db.get_message_by_id(message_id)
    assert before is not None
    before_entries = _message_sync_entries(db, message_id)
    canary = "PRIVATE-KIMI-UPDATE-CANARY"

    with pytest.raises(InputError) as invalid:
        db.update_provider_continuation(
            message_id=message_id,
            expected_message_version=before["version"],
            provider_continuation_json=_kimi_checkpoint_json(
                "different answer", post_tool=True, canary=canary
            ),
        )

    assert canary not in str(invalid.value)
    assert invalid.value.__cause__ is None
    assert invalid.value.__context__ is None
    assert db.get_message_by_id(message_id) == before
    assert _message_sync_entries(db, message_id) == before_entries

    checkpoint_json = _kimi_checkpoint_json("post-tool visible answer", post_tool=True)
    assert db.update_provider_continuation(
        message_id=message_id,
        expected_message_version=before["version"],
        provider_continuation_json=checkpoint_json,
        content="post-tool visible answer",
    )
    after = db.get_message_by_id(message_id)
    assert after is not None
    assert after["content"] == "post-tool visible answer"
    assert after["version"] == before["version"] + 1
    assert after["provider_continuation_json"] == dump_provider_continuation_json(
        parse_provider_continuation_json(checkpoint_json)
    )
    assert len(_message_sync_entries(db, message_id)) == len(before_entries) + 1


def test_clearing_image_only_owner_keeps_visible_image_and_one_update_intent(
    tmp_path: Path,
) -> None:
    db, conversation_id = _db_with_conversation(tmp_path)
    image = b"visible-image-bytes"
    message_id = db.add_message(
        {
            "id": "image-owner",
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "",
            "image_data": image,
            "image_mime_type": "image/png",
            "provider_continuation_json": _checkpoint_json(),
        }
    )
    before = db.get_message_by_id(message_id)
    assert before is not None

    assert db.update_provider_continuation(
        message_id=message_id,
        expected_message_version=before["version"],
        provider_continuation_json=None,
    )

    after = db.get_message_by_id(message_id)
    assert after is not None
    assert after["deleted"] == 0
    assert after["image_data"] == image
    assert after["image_mime_type"] == "image/png"
    assert after["provider_continuation_json"] is None
    assert after["version"] == before["version"] + 1
    entries = _message_sync_entries(db, message_id)
    assert [entry["operation"] for entry in entries] == ["create", "update"]
    assert entries[-1]["version"] == after["version"]
    assert entries[-1]["payload"]["image_mime_type"] == "image/png"
    assert entries[-1]["payload"]["provider_continuation_json"] is None


def test_clearing_attachment_only_owner_keeps_attachment_and_one_update_intent(
    tmp_path: Path,
) -> None:
    db, conversation_id = _db_with_conversation(tmp_path)
    message_id = db.add_message(
        {
            "id": "attachment-owner",
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "",
            "provider_continuation_json": _checkpoint_json(),
        }
    )
    db.set_message_attachments(
        message_id,
        [
            {
                "position": 1,
                "data": b"visible-attachment-bytes",
                "mime_type": "image/png",
                "display_name": "visible.png",
            }
        ],
    )
    before = db.get_message_by_id(message_id)
    before_attachments = db.get_attachments_for_messages([message_id])
    before_entries = _message_sync_entries(db, message_id)
    assert before is not None

    assert db.update_provider_continuation(
        message_id=message_id,
        expected_message_version=before["version"],
        provider_continuation_json=None,
    )

    after = db.get_message_by_id(message_id)
    assert after is not None
    assert after["deleted"] == 0
    assert after["provider_continuation_json"] is None
    assert after["version"] == before["version"] + 1
    assert db.get_attachments_for_messages([message_id]) == before_attachments
    entries = _message_sync_entries(db, message_id)
    assert len(entries) == len(before_entries) + 1
    assert entries[-1]["operation"] == "update"
    assert entries[-1]["version"] == after["version"]
    assert entries[-1]["payload"]["deleted"] == 0
    assert entries[-1]["payload"]["provider_continuation_json"] is None


def test_discard_and_variant_ownership_stay_on_exact_assistant_rows(
    tmp_path: Path,
) -> None:
    db, conversation_id = _db_with_conversation(tmp_path)
    blank_id = db.create_assistant_with_continuation(
        message_id="blank-owner",
        conversation_id=conversation_id,
        parent_message_id=None,
        content="",
        provider_continuation_json=_checkpoint_json(),
    )
    blank_before = db.get_message_by_id(blank_id)
    assert blank_before is not None
    assert db.update_provider_continuation(
        message_id=blank_id,
        expected_message_version=blank_before["version"],
        provider_continuation_json=None,
    )
    blank_raw = (
        db.get_connection()
        .execute(
            "SELECT deleted, version, provider_continuation_json FROM messages WHERE id = ?",
            (blank_id,),
        )
        .fetchone()
    )
    assert tuple(blank_raw) == (1, blank_before["version"] + 1, None)
    # task-19564: tombstoning the row purges its superseded content-bearing
    # `create` intent -- nothing can reach a version below a tombstone, and
    # leaving it there was how a deleted message's plaintext survived the
    # delete. The tombstone itself carries no content and is retained.
    assert [entry["operation"] for entry in _message_sync_entries(db, blank_id)] == [
        "delete",
    ]

    visible_id = db.create_assistant_with_continuation(
        message_id="visible-owner",
        conversation_id=conversation_id,
        parent_message_id=None,
        content="visible answer",
        provider_continuation_json=_checkpoint_json(),
    )
    visible_before = db.get_message_by_id(visible_id)
    assert visible_before is not None
    with pytest.raises(InputError):
        db.update_provider_continuation(
            message_id=visible_id,
            expected_message_version=visible_before["version"],
            provider_continuation_json=None,
            deleted=True,
        )
    assert db.update_provider_continuation(
        message_id=visible_id,
        expected_message_version=visible_before["version"],
        provider_continuation_json=None,
    )
    visible_after = db.get_message_by_id(visible_id)
    assert visible_after is not None
    assert visible_after["content"] == "visible answer"
    assert visible_after["deleted"] == 0
    assert visible_after["provider_continuation_json"] is None
    assert visible_after["version"] == visible_before["version"] + 1
    assert len(_message_sync_entries(db, visible_id)) == 2

    variant_id = db.create_message_variant(visible_id, "regenerated answer")
    assert variant_id is not None
    variants = {row["id"]: row for row in db.get_message_variants(visible_id)}
    assert variants[visible_id]["provider_continuation_json"] is None
    assert variants[variant_id]["provider_continuation_json"] is None

    variant_before = db.get_message_by_id(variant_id)
    assert variant_before is not None
    assert db.update_provider_continuation(
        message_id=variant_id,
        expected_message_version=variant_before["version"],
        provider_continuation_json=_checkpoint_json(canary="variant private"),
    )
    assert db.select_message_variant(variant_id)
    variants = {row["id"]: row for row in db.get_message_variants(variant_id)}
    assert variants[visible_id]["provider_continuation_json"] is None
    assert variants[variant_id]["provider_continuation_json"] is not None


def test_generic_ancestor_edit_atomically_tombstones_all_descendant_checkpoints(
    tmp_path: Path,
) -> None:
    """A content edit makes every old descendant ineligible without data loss."""
    db, conversation_id = _db_with_conversation(tmp_path)
    root_id = db.add_message(
        {
            "id": "edited-user-root",
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "original prompt",
        }
    )
    visible_id = db.create_assistant_with_continuation(
        message_id="visible-descendant",
        conversation_id=conversation_id,
        parent_message_id=root_id,
        content="visible answer",
        provider_continuation_json=_checkpoint_json(canary="VISIBLE-PRIVATE"),
    )
    variant_id = db.create_message_variant(visible_id, "off-path answer")
    assert variant_id is not None
    variant_before = db.get_message_by_id(variant_id)
    assert variant_before is not None
    db.update_provider_continuation(
        message_id=variant_id,
        expected_message_version=variant_before["version"],
        provider_continuation_json=_checkpoint_json(canary="VARIANT-PRIVATE"),
    )
    blank_id = db.create_assistant_with_continuation(
        message_id="blank-descendant",
        conversation_id=conversation_id,
        parent_message_id=visible_id,
        content="",
        provider_continuation_json=_checkpoint_json(canary="BLANK-PRIVATE"),
    )
    versions_before = {
        message_id: db.get_connection()
        .execute("SELECT version FROM messages WHERE id = ?", (message_id,))
        .fetchone()[0]
        for message_id in (root_id, visible_id, variant_id, blank_id)
    }

    assert db.update_message(
        root_id,
        {"content": "edited prompt"},
        expected_version=versions_before[root_id],
    )

    rows = {
        row["id"]: dict(row)
        for row in db.get_connection()
        .execute(
            "SELECT id, content, deleted, version, provider_continuation_json "
            "FROM messages WHERE id IN (?, ?, ?, ?)",
            (root_id, visible_id, variant_id, blank_id),
        )
        .fetchall()
    }
    assert rows[root_id]["content"] == "edited prompt"
    assert rows[root_id]["version"] == versions_before[root_id] + 1
    for message_id in (visible_id, variant_id, blank_id):
        assert rows[message_id]["provider_continuation_json"] is not None
        assert rows[message_id]["version"] == versions_before[message_id] + 1
        assert rows[message_id]["deleted"] == 1
    assert _message_sync_entries(db, visible_id)[-1]["operation"] == "delete"
    assert _message_sync_entries(db, variant_id)[-1]["operation"] == "delete"
    assert _message_sync_entries(db, blank_id)[-1]["operation"] == "delete"


def test_generic_owner_edit_clears_complete_k3_checkpoint_unless_internal_preserve(
    tmp_path: Path,
) -> None:
    db, conversation_id = _db_with_conversation(tmp_path)
    owner_id = db.create_assistant_with_continuation(
        message_id="k3-owner",
        conversation_id=conversation_id,
        parent_message_id=None,
        content="original answer",
        provider_continuation_json=_kimi_checkpoint_json("original answer"),
    )

    with pytest.raises(InputError):
        db.update_message(
            owner_id,
            {"content": "runtime contradiction"},
            expected_version=1,
            preserve_provider_continuation=True,
        )
    preserved = db.get_message_by_id(owner_id)
    assert preserved is not None
    assert preserved["version"] == 1
    assert preserved["content"] == "original answer"
    assert preserved["provider_continuation_json"] is not None

    assert db.update_message(
        owner_id,
        {"content": "user edit"},
        expected_version=preserved["version"],
    )
    edited = db.get_message_by_id(owner_id)
    assert edited is not None
    assert edited["content"] == "user edit"
    assert edited["provider_continuation_json"] is None
