"""Conversation ownership and rollback of operational mix-in records."""

import json
import sqlite3
from dataclasses import replace

import pytest

from Tests.Chat.test_console_trace_runtime import (
    make_database as make_database,  # noqa: PLC0414 - shared pytest fixture
)
from tldw_chatbook.Chat.console_prompt_mixins import (
    MixinAttemptSnapshot,
    MixinAttemptWrite,
    MixinPreferenceConflictError,
    MixinPreferences,
    MixinSelection,
    MixinSnapshot,
    MixinSnapshotUnavailableError,
    MixinSourceRef,
)
from tldw_chatbook.DB.console_prompt_mixins import (
    copy_conversation_mixins,
    read_attempt,
    read_preferences,
    save_preferences,
    write_attempt,
)
from tldw_chatbook.Prompt_Management.prompt_artifact_models import (
    LiteralMixinDefinition,
)

pytestmark = pytest.mark.bootstrap_profile


def saved_turn(db):
    cid = db.add_conversation({"title": "Mix-ins"})
    user = db.add_message(
        {"conversation_id": cid, "sender": "user", "content": "Question"}
    )
    assistant = db.add_message(
        {
            "conversation_id": cid,
            "sender": "assistant",
            "content": "Answer",
            "parent_message_id": user,
        }
    )
    return cid, {"native-user": user, "native-assistant": assistant}


def attempt(*, empty=False, attempt_id="attempt"):
    source = MixinSourceRef(
        "local:profile", "local", "d85e8026-7217-45a9-8515-2ab663205741"
    )
    snapshot = MixinSnapshot(
        source, 7, "PRIVATE NAME", LiteralMixinDefinition("PRIVATE BODY\n", "before")
    )
    return MixinAttemptWrite(
        attempt_id,
        "native-user",
        "native-assistant",
        MixinAttemptSnapshot(() if empty else (snapshot,)),
    )


def test_preference_write_rolls_back_with_parent_transaction(make_database):
    db = make_database(":memory:", "mixin-test")
    cid, _ = saved_turn(db)
    with pytest.raises(RuntimeError), db.transaction() as cursor:
        save_preferences(cursor, cid, expected_revision=0, value=MixinPreferences())
        raise RuntimeError("abort parent")
    with db.transaction() as cursor:
        assert read_preferences(cursor, cid).status == "absent"


def test_preferences_compare_revision_and_preserve_future_content(make_database):
    db = make_database(":memory:", "mixin-test")
    cid, _ = saved_turn(db)
    item = attempt().snapshot.items[0]
    value = MixinPreferences(
        entries=(MixinSelection(item.source, False, item.name, "before", 7),)
    )
    with db.transaction() as cursor:
        saved = save_preferences(cursor, cid, expected_revision=0, value=value)
        assert saved.revision == 1
        assert read_preferences(cursor, cid).preferences == saved
        with pytest.raises(MixinPreferenceConflictError):
            save_preferences(cursor, cid, expected_revision=0, value=value)
        future = '{"schema_version":2,"future":"PRIVATE"}'
        cursor.execute(
            "UPDATE console_prompt_mixin_preferences SET schema_version=2, selection_json=? WHERE conversation_id=?",
            (future, cid),
        )
        assert read_preferences(cursor, cid).status == "unsupported"
        with pytest.raises(MixinPreferenceConflictError):
            save_preferences(cursor, cid, expected_revision=1, value=value)
        assert read_preferences(cursor, cid).raw == future


def test_attempt_rollback_deduplication_immutability_and_missing_items(make_database):
    db = make_database(":memory:", "mixin-test")
    cid, ids = saved_turn(db)
    batch = attempt()
    with pytest.raises(RuntimeError), db.transaction() as cursor:
        write_attempt(cursor, cid, batch=batch, native_message_ids=ids)
        raise RuntimeError("abort")
    with db.transaction() as cursor:
        assert read_attempt(cursor, cid, attempt_id=batch.attempt_id).status == "legacy"
        write_attempt(cursor, cid, batch=batch, native_message_ids=ids)
        write_attempt(cursor, cid, batch=batch, native_message_ids=ids)
        assert (
            read_attempt(cursor, cid, attempt_id=batch.attempt_id).snapshot
            == batch.snapshot
        )
        with pytest.raises(MixinSnapshotUnavailableError):
            write_attempt(
                cursor, cid, batch=attempt(empty=True), native_message_ids=ids
            )
        with pytest.raises(sqlite3.IntegrityError):
            cursor.execute("UPDATE console_prompt_mixin_revisions SET body='changed'")
        cursor.execute("DELETE FROM console_prompt_mixin_attempt_items")
        assert (
            read_attempt(cursor, cid, attempt_id=batch.attempt_id).status
            == "unavailable"
        )


def test_empty_attempt_and_lost_marked_record_are_distinct(make_database):
    db = make_database(":memory:", "mixin-test")
    cid, ids = saved_turn(db)
    with db.transaction() as cursor:
        write_attempt(cursor, cid, batch=attempt(empty=True), native_message_ids=ids)
        result = read_attempt(cursor, cid, attempt_id="attempt")
        assert result.status == "recorded" and result.snapshot == MixinAttemptSnapshot()
        raw = cursor.execute(
            "SELECT metadata_json FROM messages WHERE id=?", (ids["native-assistant"],)
        ).fetchone()[0]
        assert json.loads(raw)["mixin_snapshot_schema_version"] == 1
        assert json.loads(raw)["mixin_attempt_id"] == "attempt"
        cursor.execute(
            "DELETE FROM console_prompt_mixin_attempts WHERE conversation_id=?", (cid,)
        )
        assert read_attempt(cursor, cid, attempt_id="attempt").status == "unavailable"


def test_foreign_anchors_refused_and_fork_survives_source_purge(make_database):
    db = make_database(":memory:", "mixin-test")
    cid, ids = saved_turn(db)
    other, other_ids = saved_turn(db)
    with db.transaction() as cursor:
        with pytest.raises(MixinSnapshotUnavailableError):
            write_attempt(cursor, other, batch=attempt(), native_message_ids=ids)
        write_attempt(cursor, cid, batch=attempt(), native_message_ids=ids)
        save_preferences(cursor, cid, expected_revision=0, value=MixinPreferences())
        copy_conversation_mixins(
            cursor,
            source_conversation_id=cid,
            destination_conversation_id=other,
            message_ids={ids[k]: other_ids[k] for k in ids},
        )
        assert read_preferences(cursor, other).status == "absent"
    db.soft_delete_conversation(cid, expected_version=1)
    with db.transaction() as cursor:
        assert read_attempt(cursor, cid, attempt_id="attempt").status == "recorded"
    from tldw_chatbook.Canvas.repository import CanvasRepository

    CanvasRepository(db).hard_purge_conversation(cid)
    with db.transaction() as cursor:
        assert (
            read_attempt(cursor, other, attempt_id="attempt").snapshot
            == attempt().snapshot
        )
        assert cursor.execute("PRAGMA foreign_key_check").fetchall() == []
        matches = cursor.execute(
            "SELECT id FROM conversations WHERE rowid IN (SELECT rowid FROM conversations_fts WHERE conversations_fts MATCH 'Mix')"
        ).fetchall()
        assert [row[0] for row in matches] == [other]
        cursor.execute(
            "INSERT INTO conversations_fts(conversations_fts) VALUES('integrity-check')"
        )
    db.soft_delete_conversation(other, expected_version=1)
    CanvasRepository(db).hard_purge_conversation(other)
    with db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO conversations_fts(conversations_fts) VALUES('integrity-check')"
        )
        assert (
            cursor.execute(
                "SELECT COUNT(*) FROM console_prompt_mixin_revisions"
            ).fetchone()[0]
            == 0
        )


def test_hard_message_delete_keeps_attempt_without_anchor(make_database):
    db = make_database(":memory:", "mixin-test")
    cid, ids = saved_turn(db)
    with db.transaction() as cursor:
        write_attempt(cursor, cid, batch=attempt(), native_message_ids=ids)
    from tldw_chatbook.Chat.console_semantic_revision import SemanticRevisionCoordinator

    with db.transaction() as cursor:
        SemanticRevisionCoordinator(db).mutate_message(
            cursor,
            message_id=ids["native-assistant"],
            creation_reason="message_delete",
            hard_delete=True,
        )
    with db.transaction() as cursor:
        assert read_attempt(cursor, cid, attempt_id="attempt").status == "recorded"


def test_writer_lifetime_and_exact_promotion_anchors(make_database):
    from tldw_chatbook.Chat.console_prompt_mixin_service import MixinAttemptContribution
    from tldw_chatbook.Chat.console_transaction_contribution import (
        _scoped_console_transaction_writer,
    )

    db = make_database(":memory:", "mixin-test")
    cid, ids = saved_turn(db)
    batch = replace(attempt(), user_message_id="assistant", assistant_message_id="user")
    mapping = {"assistant": ids["native-user"], "user": ids["native-assistant"]}
    contribution = MixinAttemptContribution(batch)
    with db.transaction() as cursor:
        with _scoped_console_transaction_writer(cursor, cid) as writer:
            contribution.write_exact(
                writer=writer, conversation_id=cid, native_message_ids=mapping
            )
        with pytest.raises(RuntimeError, match="active contribution"):
            writer.append_mixin_attempt(batch=batch, native_message_ids=mapping)
        assert (
            read_attempt(cursor, cid, attempt_id="attempt").snapshot == batch.snapshot
        )
    assert "PRIVATE" not in repr(contribution)


def test_acceptance_uses_explicit_native_map_and_rolls_back(make_database):
    from Tests.Chat.test_console_transaction_contribution import _acceptance
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_prompt_mixin_service import MixinAttemptContribution

    db = make_database(":memory:", "mixin-test")
    cid = db.add_conversation({"title": "Acceptance"})
    batch = replace(attempt(), user_message_id="assistant", assistant_message_id="user")
    service = ChatPersistenceService(db)
    acceptance = replace(
        _acceptance(cid, MixinAttemptContribution(batch)),
        user_message_id="assistant",
        assistant_message_id="user",
    )
    for patch in (
        {"user_message_id": "unknown-user"},
        {"assistant_message_id": "unknown-assistant"},
    ):
        with pytest.raises(ValueError), db.transaction() as cursor:
            service.console_dispatch_repository.insert_with_messages(
                cursor, replace(acceptance, **patch)
            )
        assert db.get_message_by_id("assistant") is None
    with pytest.raises(RuntimeError), db.transaction() as cursor:
        service.console_dispatch_repository.insert_with_messages(cursor, acceptance)
        raise RuntimeError("abort acceptance")
    with db.transaction() as cursor:
        assert read_attempt(cursor, cid, attempt_id="attempt").status == "legacy"
        service.console_dispatch_repository.insert_with_messages(cursor, acceptance)
        assert (
            read_attempt(cursor, cid, attempt_id="attempt").snapshot == batch.snapshot
        )
        row = cursor.execute(
            "SELECT user_message_id,assistant_message_id FROM console_prompt_mixin_attempts"
        ).fetchone()
        assert tuple(row) == ("assistant", "user")


def test_frozen_contribution_identity_includes_literal_body():
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_prompt_mixin_service import MixinAttemptContribution

    original = MixinAttemptContribution(attempt())
    changed = MixinAttemptContribution(attempt(empty=True))
    assert ConsoleChatStore._contribution_fingerprint(
        original
    ) != ConsoleChatStore._contribution_fingerprint(changed)


def test_native_promotion_writes_temporary_attempt_once(make_database):
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_library_policy import (
        ConsoleAssistantLibraryAccess,
        ConsoleAutoRetrieve,
        ConsoleLibraryPolicyCandidate,
    )
    from tldw_chatbook.Chat.console_prompt_mixin_service import MixinAttemptContribution

    db = make_database(":memory:", "mixin-test")
    service = ChatPersistenceService(db)
    batch = replace(attempt(), user_message_id="assistant", assistant_message_id="user")
    contribution = MixinAttemptContribution(batch)
    service.promote_console_conversation_bundle(
        conversation_id="promoted",
        conversation_kwargs={"conversation_title": "Temporary"},
        policy_candidate=ConsoleLibraryPolicyCandidate(
            auto_retrieve=ConsoleAutoRetrieve.NEVER,
            assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
        ),
        messages=(
            {
                "native_id": "assistant",
                "create_kwargs": {
                    "sender": "user",
                    "content": "Question",
                    "message_id": "promoted-user",
                },
            },
            {
                "native_id": "user",
                "create_kwargs": {
                    "sender": "assistant",
                    "content": "Answer",
                    "message_id": "promoted-assistant",
                    "parent_message_id": "promoted-user",
                },
            },
        ),
        active_leaf_message_id="promoted-assistant",
        contributions=(contribution, contribution),
    )
    with db.transaction() as cursor:
        assert (
            read_attempt(cursor, "promoted", attempt_id="attempt").snapshot
            == batch.snapshot
        )
        assert (
            cursor.execute(
                "SELECT COUNT(*) FROM console_prompt_mixin_attempts"
            ).fetchone()[0]
            == 1
        )
        assert read_preferences(cursor, "promoted").status == "absent"
        cursor.execute(
            "DELETE FROM console_prompt_mixin_attempts WHERE conversation_id='promoted'"
        )
        assert (
            read_attempt(cursor, "promoted", attempt_id="attempt").status
            == "unavailable"
        )


def test_integrated_fork_copies_receipts_and_hydrates_markers(make_database):
    from Tests.Chat.test_console_chat_fork_persistence import (
        _commit,
        _png_bytes,
        _snapshot,
    )
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_chat_fork import ConsoleForkProjectedAttachment
    from tldw_chatbook.Chat.console_conversation_hydration import (
        console_messages_from_conversation_tree,
    )

    db = make_database(":memory:", "mixin-test")
    snapshot = _snapshot(db)
    projected = snapshot.messages[1]
    snapshot = replace(
        snapshot,
        messages=(
            snapshot.messages[0],
            replace(
                projected,
                status="stopped",
                trace_turn_id="source-trace-alias",
                attachments=(
                    ConsoleForkProjectedAttachment(
                        owner_native_message_id=projected.native_message_id,
                        owner_persisted_message_id=projected.persisted_message_id,
                        position=0,
                        data=_png_bytes((255, 0, 0)),
                        mime_type="image/png",
                        display_name="retained.png",
                    ),
                ),
            ),
        ),
    )
    with db.transaction() as cursor:
        write_attempt(
            cursor,
            "source",
            batch=attempt(),
            native_message_ids={
                "native-user": "source-user",
                "native-assistant": "source-assistant",
            },
        )
    service = ChatPersistenceService(db)
    result = _commit(service, snapshot)
    with db.transaction() as cursor:
        assert (
            read_attempt(cursor, result.conversation_id, attempt_id="attempt").snapshot
            == attempt().snapshot
        )
        row = cursor.execute(
            "SELECT metadata_json FROM messages WHERE id='fork-assistant'"
        ).fetchone()[0]
        from tldw_chatbook.Chat.message_metadata import MessageMetadata

        assert MessageMetadata.from_json(row).mixin_attempt_id == "attempt"
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService

    tree = ChatConversationService(db).get_conversation_tree(result.conversation_id)
    messages = console_messages_from_conversation_tree(tree, db=db)
    assistant = next(
        item for item in messages if item.persisted_message_id == "fork-assistant"
    )
    assert assistant.status == "stopped"
    assert assistant.attachment_label == "retained.png"
    assert assistant.attachments[0].display_name == "retained.png"
    assert assistant.trace_turn_id == "source-trace-alias"
    assert assistant.metadata.mixin_snapshot_schema_version == 1
    assert assistant.metadata.mixin_attempt_id == "attempt"
    with db.transaction() as cursor:
        assert read_preferences(cursor, result.conversation_id).status == "absent"


def test_checkpoint_marker_survives_message_projection_loss(make_database):
    from Tests.Chat.test_console_transaction_contribution import _acceptance
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_prompt_mixin_service import MixinAttemptContribution

    db = make_database(":memory:", "mixin-test")
    cid = db.add_conversation({"title": "Checkpoint"})
    batch = replace(
        attempt(empty=True), user_message_id="user", assistant_message_id="assistant"
    )
    service = ChatPersistenceService(db)
    with db.transaction() as cursor:
        result = service.console_dispatch_repository.insert_with_messages(
            cursor, _acceptance(cid, MixinAttemptContribution(batch))
        )
        assert result.mixin_snapshot_schema_version == 1
        cursor.execute("UPDATE messages SET metadata_json=NULL WHERE id='assistant'")
        cursor.execute(
            "DELETE FROM console_prompt_mixin_attempts WHERE conversation_id=?", (cid,)
        )
        assert read_attempt(cursor, cid, attempt_id="attempt").status == "unavailable"


def test_terminal_metadata_rewrite_retains_marker_and_fork_unavailability(
    make_database,
):
    from tldw_chatbook.Chat.message_metadata import MessageMetadata

    db = make_database(":memory:", "mixin-test")
    cid, ids = saved_turn(db)
    other, other_ids = saved_turn(db)
    with db.transaction() as cursor:
        write_attempt(cursor, cid, batch=attempt(), native_message_ids=ids)
    assert db.update_message_metadata_local(
        ids["native-assistant"], MessageMetadata(provider="test").to_json()
    )
    with db.transaction() as cursor:
        cursor.execute(
            "DELETE FROM console_prompt_mixin_attempts WHERE conversation_id=?", (cid,)
        )
        assert read_attempt(cursor, cid, attempt_id="attempt").status == "unavailable"
        copy_conversation_mixins(
            cursor,
            source_conversation_id=cid,
            destination_conversation_id=other,
            message_ids={ids[k]: other_ids[k] for k in ids},
        )
        assert read_attempt(cursor, other, attempt_id="attempt").status == "unavailable"


def test_snapshot_reuse_retry_link_and_cross_owner_fk(make_database):
    db = make_database(":memory:", "mixin-test")
    cid, ids = saved_turn(db)
    other, foreign = saved_turn(db)
    second_assistant = db.add_message(
        {
            "conversation_id": cid,
            "sender": "assistant",
            "content": "Retry",
            "parent_message_id": ids["native-user"],
        }
    )
    batch = attempt()
    retry = replace(
        batch,
        attempt_id="retry",
        assistant_message_id="retry-assistant",
        retry_of_attempt_id=batch.attempt_id,
    )
    with db.transaction() as cursor:
        write_attempt(cursor, cid, batch=batch, native_message_ids=ids)
        write_attempt(
            cursor,
            cid,
            batch=retry,
            native_message_ids={**ids, "retry-assistant": second_assistant},
        )
        assert read_attempt(cursor, cid, attempt_id="retry").snapshot == batch.snapshot
        assert (
            cursor.execute(
                "SELECT COUNT(*) FROM console_prompt_mixin_revisions"
            ).fetchone()[0]
            == 1
        )
        snapshot_id = cursor.execute(
            "SELECT snapshot_id FROM console_prompt_mixin_revisions"
        ).fetchone()[0]
        write_attempt(
            cursor, other, batch=attempt(empty=True), native_message_ids=foreign
        )
        with pytest.raises(sqlite3.IntegrityError):
            cursor.execute(
                "INSERT INTO console_prompt_mixin_attempt_items VALUES(?,?,?,?)",
                (other, "attempt", 0, snapshot_id),
            )
        row = cursor.execute(
            "SELECT retry_of_attempt_id FROM console_prompt_mixin_attempts WHERE attempt_id='retry'"
        ).fetchone()
        assert row[0] == "attempt"
        cursor.execute(
            "DELETE FROM console_prompt_mixin_attempt_items WHERE conversation_id=? AND attempt_id='retry'",
            (cid,),
        )
        assert read_attempt(cursor, cid, attempt_id="retry").status == "unavailable"
        assert read_attempt(cursor, cid, attempt_id="attempt").status == "recorded"


@pytest.mark.parametrize(
    "replacement",
    ['{"provider":"replacement"}', None, "null", "[]", '"metadata"', "42", "true"],
)
def test_terminal_settlement_reopen_and_missing_record_refusal(
    make_database, tmp_path, replacement
):
    from Tests.Chat.test_console_transaction_contribution import _acceptance
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_dispatch_checkpoint import (
        ConsoleAssistantSettlement,
        ConsoleDispatchCheckpointState,
        ConsoleDispatchResultStatus,
    )
    from tldw_chatbook.Chat.console_prompt_mixin_service import MixinAttemptContribution
    from tldw_chatbook.Chat.message_metadata import MessageMetadata

    path = tmp_path / "settled.sqlite"
    db = make_database(path, "mixin-test")
    cid = db.add_conversation({"title": "Checkpoint"})
    batch = replace(attempt(), user_message_id="user", assistant_message_id="assistant")
    service = ChatPersistenceService(db)
    with db.transaction() as cursor:
        service.console_dispatch_repository.insert_with_messages(
            cursor, _acceptance(cid, MixinAttemptContribution(batch))
        )
    result = service.console_dispatch_repository.settle_with_assistant(
        ConsoleAssistantSettlement(
            assistant_message_id="assistant",
            expected_checkpoint_state=ConsoleDispatchCheckpointState.ACCEPTED,
            expected_checkpoint_revision=1,
            expected_user_message_version=1,
            expected_assistant_message_version=1,
            terminal_state="stopped",
            content="Stopped",
            metadata_json=MessageMetadata(provider="test").to_json(),
        )
    )
    assert result.status == ConsoleDispatchResultStatus.COMMITTED
    if replacement is None:
        with db.transaction() as cursor:
            cursor.execute(
                "UPDATE messages SET metadata_json=NULL WHERE id='assistant'"
            )
    else:
        assert db.update_message_metadata_local("assistant", replacement)
    db.close()
    reopened = make_database(path, "mixin-reopened")
    with reopened.transaction() as cursor:
        assert (
            read_attempt(cursor, cid, attempt_id="attempt").snapshot == batch.snapshot
        )
        assert (
            cursor.execute(
                "SELECT COUNT(*) FROM console_dispatch_checkpoints"
            ).fetchone()[0]
            == 0
        )
        cursor.execute(
            "DELETE FROM console_prompt_mixin_attempts WHERE conversation_id=?", (cid,)
        )
        assert read_attempt(cursor, cid, attempt_id="attempt").status == "unavailable"


def test_legacy_fork_ignores_unrelated_malformed_metadata(make_database):
    db = make_database(":memory:", "mixin-test")
    cid, ids = saved_turn(db)
    other, other_ids = saved_turn(db)
    with db.transaction() as cursor:
        cursor.execute(
            "UPDATE messages SET metadata_json='legacy garbage' WHERE id=?",
            (ids["native-assistant"],),
        )
        copy_conversation_mixins(
            cursor,
            source_conversation_id=cid,
            destination_conversation_id=other,
            message_ids={ids[k]: other_ids[k] for k in ids},
        )
        assert read_attempt(cursor, other, attempt_id="legacy").status == "legacy"
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService

    assert (
        ChatPersistenceService(db)
        .get_conversation_mixin_attempt(
            other, assistant_message_id=other_ids["native-assistant"]
        )
        .status
        == "unavailable"
    )


@pytest.mark.parametrize(
    "companions, valid",
    [
        ({"mixin_attempt_id": "attempt", "mixin_snapshot_schema_version": 1}, True),
        ({"mixin_attempt_id": "attempt"}, False),
        ({"mixin_snapshot_schema_version": 1}, False),
        ({"mixin_attempt_id": "", "mixin_snapshot_schema_version": 1}, False),
        ({"mixin_attempt_id": 1, "mixin_snapshot_schema_version": 1}, False),
        ({"mixin_attempt_id": "attempt", "mixin_snapshot_schema_version": True}, False),
        ({"mixin_attempt_id": "attempt", "mixin_snapshot_schema_version": 2}, False),
        (
            {
                "mixin_attempt_id": "attempt",
                "mixin_snapshot_schema_version": 1,
                "authority": "execute",
            },
            False,
        ),
    ],
)
def test_fork_codec_only_accepts_exact_inert_mixin_companions(companions, valid):
    from tldw_chatbook.Chat.console_chat_fork import (
        encode_console_fork_message_metadata,
        parse_console_fork_message_metadata,
    )

    payload = json.loads(
        encode_console_fork_message_metadata("failed", "retained.png", "trace-alias")
    )
    payload.update(companions)
    assert parse_console_fork_message_metadata(json.dumps(payload)) == (
        ("failed", "retained.png", "trace-alias") if valid else None
    )


def test_contribution_digest_supports_full_body_limit_and_all_batch_fields():
    from tldw_chatbook.Chat.console_prompt_mixin_service import MixinAttemptContribution

    original = attempt()
    item = original.snapshot.items[0]
    large = replace(
        original,
        snapshot=MixinAttemptSnapshot(
            (replace(item, definition=LiteralMixinDefinition("x" * 20000, "before")),)
        ),
    )
    digest = MixinAttemptContribution(large).durable_acceptance_fingerprint()
    assert len(digest) == 64 and "xxxx" not in repr(MixinAttemptContribution(large))
    for changed in (
        replace(large, attempt_id="another"),
        replace(large, user_message_id="another"),
        replace(large, assistant_message_id="another"),
        replace(large, retry_of_attempt_id="previous"),
        replace(
            large,
            snapshot=MixinAttemptSnapshot(
                (replace(large.snapshot.items[0], name="changed"),)
            ),
        ),
    ):
        assert (
            MixinAttemptContribution(changed).durable_acceptance_fingerprint() != digest
        )


@pytest.mark.parametrize("empty", [False, True])
def test_same_row_retry_marker_transition_and_parent_rollback(make_database, empty):
    db = make_database(":memory:", "mixin-test")
    cid, ids = saved_turn(db)
    first = attempt(empty=empty)
    retry = replace(first, attempt_id="retry", retry_of_attempt_id=first.attempt_id)
    with db.transaction() as cursor:
        write_attempt(cursor, cid, batch=first, native_message_ids=ids)
    with pytest.raises(RuntimeError), db.transaction() as cursor:
        write_attempt(cursor, cid, batch=retry, native_message_ids=ids)
        raise RuntimeError("forced rollback")
    with db.transaction() as cursor:
        marker = lambda: json.loads(
            cursor.execute(
                "SELECT metadata_json FROM messages WHERE id=?",
                (ids["native-assistant"],),
            ).fetchone()[0]
        )["mixin_attempt_id"]
        assert marker() == first.attempt_id
        assert (
            cursor.execute(
                "SELECT count(*) FROM console_prompt_mixin_attempts"
            ).fetchone()[0]
            == 1
        )
        write_attempt(cursor, cid, batch=retry, native_message_ids=ids)
        assert marker() == retry.attempt_id
        assert (
            read_attempt(cursor, cid, attempt_id=first.attempt_id).snapshot
            == first.snapshot
        )
        cursor.execute(
            "UPDATE messages SET metadata_json=? WHERE id=?",
            (
                '{"mixin_attempt_id":"forged","mixin_snapshot_schema_version":1}',
                ids["native-assistant"],
            ),
        )
        assert marker() == retry.attempt_id


@pytest.mark.parametrize("empty", [False, True])
def test_copy_same_row_retry_chain_publishes_exact_marker_once(make_database, empty):
    db = make_database(":memory:", "mixin-test")
    source, source_ids = saved_turn(db)
    target, target_ids = saved_turn(db)
    batches = [attempt(empty=empty)]
    for index in range(3):
        batches.append(
            replace(
                batches[-1],
                attempt_id=f"retry-{index}",
                retry_of_attempt_id=batches[-1].attempt_id,
            )
        )
    with db.transaction() as cursor:
        for batch in batches:
            write_attempt(cursor, source, batch=batch, native_message_ids=source_ids)
        copy_conversation_mixins(
            cursor,
            source_conversation_id=source,
            destination_conversation_id=target,
            message_ids={source_ids[k]: target_ids[k] for k in source_ids},
        )
        assert read_preferences(cursor, target).status == "absent"
    from tldw_chatbook.Canvas.repository import CanvasRepository

    db.soft_delete_conversation(source, expected_version=1)
    CanvasRepository(db).hard_purge_conversation(source)
    with db.transaction() as cursor:
        marker = json.loads(
            cursor.execute(
                "SELECT metadata_json FROM messages WHERE id=?",
                (target_ids["native-assistant"],),
            ).fetchone()[0]
        )
        assert marker["mixin_attempt_id"] == batches[-1].attempt_id
        for batch in batches:
            assert (
                read_attempt(cursor, target, attempt_id=batch.attempt_id).snapshot
                == batch.snapshot
            )
        assert cursor.execute(
            "SELECT count(*) FROM console_prompt_mixin_revisions WHERE conversation_id=?",
            (target,),
        ).fetchone()[0] == (0 if empty else 1)
        assert cursor.execute("PRAGMA foreign_key_check").fetchall() == []


def test_mixin_indexes_service_real_lookup_and_message_cleanup_without_statistics(
    make_database,
):
    from tldw_chatbook.DB.console_prompt_mixins import _SNAPSHOT_COLUMNS

    db = make_database(":memory:", "mixin-plans")
    cid, first_ids = saved_turn(db)
    with db.transaction() as cursor:
        for number in range(200):
            batch = attempt(attempt_id=f"plan-{number}")
            if number:
                batch = replace(
                    batch,
                    retry_of_attempt_id=f"plan-{number - 1}",
                    replaces_attempt_id=f"plan-{number - 1}",
                )
            item = replace(batch.snapshot.items[0], record_version=number + 1)
            write_attempt(
                cursor,
                cid,
                batch=replace(batch, snapshot=MixinAttemptSnapshot((item,))),
                native_message_ids=first_ids,
            )
        assert (
            cursor.execute(
                "SELECT 1 FROM sqlite_master WHERE name='sqlite_stat1'"
            ).fetchone()
            is None
        )
        digest = cursor.execute(
            "SELECT digest FROM console_prompt_mixin_revisions LIMIT 1"
        ).fetchone()[0]
        lookup = cursor.execute(
            "EXPLAIN QUERY PLAN SELECT snapshot_id,"
            + _SNAPSHOT_COLUMNS
            + " FROM console_prompt_mixin_revisions WHERE conversation_id=? AND digest=?",
            (cid, digest),
        ).fetchall()
        assert any("idx_console_mixin_revision_digest" in row[3] for row in lookup), (
            lookup
        )
        # Real parent cleanup runs SQLite's SET NULL child searches for both
        # live message anchors; no INDEXED BY or planner statistics are supplied.
        cleanup = cursor.execute(
            "EXPLAIN QUERY PLAN DELETE FROM messages WHERE id=?",
            (first_ids["native-user"],),
        ).fetchall()
        for index in (
            "idx_console_mixin_attempt_user",
            "idx_console_mixin_attempt_assistant",
        ):
            assert any(index in row[3] for row in cleanup), (index, cleanup)
        assert cursor.execute("PRAGMA foreign_key_check").fetchall() == []
