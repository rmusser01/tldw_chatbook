from contextlib import contextmanager
from uuid import uuid4

import pytest

import tldw_chatbook.Chat.chat_persistence_service as chat_persistence_module
from tldw_chatbook.Chat.chat_persistence_service import (
    ChatPersistenceService,
)
from tldw_chatbook.Chat.console_voice_promotion import (
    ConsoleSessionBindingOrigin,
    ResolvedVoicePromotionDestination,
    VoicePromotionContext,
    derive_voice_promotion_identities,
)
from tldw_chatbook.Chat.conversation_local_marks_service import (
    ConversationLocalMarksService,
)
from tldw_chatbook.Chat.message_metadata import MessageMetadata
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService


@pytest.fixture
def client_id():
    return "test_chat_persistence_client"


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_chat_persistence.sqlite"


@pytest.fixture
def db_instance(db_path, client_id):
    db = CharactersRAGDB(db_path, client_id)
    yield db
    db.close_connection()


_VOICE_PROMOTION_ID = "voice-promotion-a"
_OTHER_TERMINAL_RECEIPT = "44444444-4444-4444-8444-444444444444"
_VOICE_USER_TEXT = "exact private voice transcript"
_VOICE_ASSISTANT_TEXT = "exact private completed reply"
_VOICE_USAGE_JSON = ProviderUsage(
    uncached_input=7,
    output=11,
    provider="openai",
    model="gpt-voice-test",
).to_json()


def _voice_promotion_case(
    db: CharactersRAGDB,
    *,
    promotion_id: str = _VOICE_PROMOTION_ID,
) -> tuple[str, str, ResolvedVoicePromotionDestination, VoicePromotionContext]:
    conversation_id = db.add_conversation(
        {"title": "Voice pair target", "character_id": None}
    )
    root_message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "existing durable leaf",
            "id": "voice-existing-leaf",
        }
    )
    assert root_message_id == "voice-existing-leaf"
    db.set_conversation_active_leaf(conversation_id, root_message_id)
    origin = ConsoleSessionBindingOrigin(
        session_id="voice-session",
        session_incarnation=1,
        persisted_conversation_id=conversation_id,
        conversation_binding_revision=3,
    )
    context = VoicePromotionContext(
        promotion_id=promotion_id,
        attempt_id="voice-attempt",
        origin=origin,
        expected_native_leaf_id="native-existing-leaf",
        expected_persisted_leaf_id=root_message_id,
        user_text=_VOICE_USER_TEXT,
        assistant_text=_VOICE_ASSISTANT_TEXT,
        usage_json=_VOICE_USAGE_JSON,
        terminal_boundary_id="terminal-boundary",
        capture_eligible_at_dispatch=True,
    )
    destination = ResolvedVoicePromotionDestination(
        session_id=origin.session_id,
        session_incarnation=origin.session_incarnation,
        persisted_conversation_id=conversation_id,
        expected_persisted_leaf_id=root_message_id,
        capture_eligible_at_dispatch=True,
    )
    return conversation_id, root_message_id, destination, context


def _promotion_identity_rows(
    db: CharactersRAGDB,
    *,
    promotion_id: str = _VOICE_PROMOTION_ID,
) -> tuple[list[object], list[object]]:
    identities = derive_voice_promotion_identities(promotion_id)
    connection = db.get_connection()
    messages = connection.execute(
        "SELECT * FROM messages WHERE id IN (?, ?)",
        (identities.user_message_id, identities.assistant_message_id),
    ).fetchall()
    marks = connection.execute(
        """SELECT conversation_id, mark_type
             FROM conversation_local_marks
            WHERE mark_type IN (?, ?, ?)""",
        (
            ConversationLocalMarksService.console_unseen_mark_type(
                identities.terminal_receipt_id
            ),
            ConversationLocalMarksService.console_terminal_outcome_mark_type(
                identities.terminal_receipt_id, "complete"
            ),
            ConversationLocalMarksService.console_terminal_outcome_mark_type(
                identities.terminal_receipt_id, "failed"
            ),
        ),
    ).fetchall()
    return messages, marks


def _promotion_persistence_snapshot(
    db: CharactersRAGDB,
    *,
    conversation_id: str,
    promotion_id: str = _VOICE_PROMOTION_ID,
) -> dict[str, object]:
    identities = derive_voice_promotion_identities(promotion_id)
    message_ids = (identities.user_message_id, identities.assistant_message_id)
    connection = db.get_connection()

    def frozen_rows(rows) -> tuple[tuple[object, ...], ...]:
        return tuple(sorted((tuple(row) for row in rows), key=repr))

    messages = frozen_rows(
        connection.execute(
            "SELECT * FROM messages WHERE id IN (?, ?)", message_ids
        ).fetchall()
    )
    revisions = connection.execute(
        """SELECT * FROM console_trace_semantic_revisions
            WHERE source_message_id IN (?, ?)
               OR live_message_id IN (?, ?)""",
        message_ids * 2,
    ).fetchall()
    revision_ids = tuple(row[0] for row in revisions)
    sidecars: dict[tuple[str, str], tuple[tuple[object, ...], ...]] = {}
    for table, column in sorted(
        chat_persistence_module._VOICE_PROMOTION_FORBIDDEN_MESSAGE_LOCATORS
    ):
        sidecars[(table, column)] = frozen_rows(
            connection.execute(
                f'SELECT * FROM "{table}" WHERE "{column}" IN (?, ?)',
                message_ids,
            ).fetchall()
        )
    if revision_ids:
        placeholders = ", ".join("?" for _revision_id in revision_ids)
        for table, column in sorted(
            chat_persistence_module._VOICE_PROMOTION_FORBIDDEN_REVISION_LOCATORS
        ):
            sidecars[(table, column)] = frozen_rows(
                connection.execute(
                    f'SELECT * FROM "{table}" WHERE "{column}" IN ({placeholders})',
                    revision_ids,
                ).fetchall()
            )
    marks = frozen_rows(
        connection.execute(
            """SELECT conversation_id, mark_type,
                      CAST(created_at AS TEXT), CAST(updated_at AS TEXT)
                 FROM conversation_local_marks
                WHERE mark_type = ? OR mark_type LIKE ?""",
            (
                ConversationLocalMarksService.console_unseen_mark_type(
                    identities.terminal_receipt_id
                ),
                (
                    f"{ConversationLocalMarksService.CONSOLE_TERMINAL_OUTCOME_PREFIX}"
                    f"{identities.terminal_receipt_id}:%"
                ),
            ),
        ).fetchall()
    )
    active_leaf = connection.execute(
        "SELECT active_leaf_message_id FROM conversations WHERE id = ?",
        (conversation_id,),
    ).fetchone()
    return {
        "messages": messages,
        "revisions": frozen_rows(revisions),
        "sidecars": sidecars,
        "terminal_marks": marks,
        "active_leaf": None if active_leaf is None else active_leaf[0],
    }


def test_voice_promotion_locator_inventory_covers_owned_message_revision_tables(
    db_instance,
):
    connection = db_instance.get_connection()
    discovered: set[tuple[str, str]] = set()
    tables = tuple(
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        if not row[0].startswith("sqlite_")
    )
    for table in tables:
        columns = connection.execute(f'PRAGMA table_info("{table}")').fetchall()
        discovered.update(
            (table, column[1])
            for column in columns
            if column[1] in {"message_id", "revision_id"}
            or column[1].endswith("_message_id")
            or column[1].endswith("_revision_id")
        )
        foreign_keys = connection.execute(
            f'PRAGMA foreign_key_list("{table}")'
        ).fetchall()
        discovered.update(
            (table, foreign_key[3])
            for foreign_key in foreign_keys
            if (foreign_key[2], foreign_key[4])
            in {
                ("messages", "id"),
                ("console_trace_semantic_revisions", "revision_id"),
            }
        )

    categories = (
        getattr(
            chat_persistence_module,
            "_VOICE_PROMOTION_MANDATORY_LOCATORS",
            frozenset(),
        ),
        getattr(
            chat_persistence_module,
            "_VOICE_PROMOTION_FORBIDDEN_MESSAGE_LOCATORS",
            frozenset(),
        ),
        getattr(
            chat_persistence_module,
            "_VOICE_PROMOTION_FORBIDDEN_REVISION_LOCATORS",
            frozenset(),
        ),
        getattr(
            chat_persistence_module,
            "_VOICE_PROMOTION_LOCATOR_EXEMPTIONS",
            frozenset(),
        ),
    )
    categorized = frozenset().union(*categories)

    assert sum(map(len, categories)) == len(categorized)
    assert discovered == categorized


@pytest.mark.parametrize("already_committed", [False, True])
def test_archive_blocks_new_voice_pair_but_preserves_committed_reconciliation(
    db_instance, already_committed
):
    conversation_id, _, destination, context = _voice_promotion_case(db_instance)
    service = ChatPersistenceService(db_instance)
    if already_committed:
        service.commit_completed_voice_pair(destination=destination, context=context)
    row = db_instance.get_conversation_by_id(conversation_id)
    db_instance.set_conversations_archived(
        [conversation_id],
        archived=True,
        expected_versions={conversation_id: row["version"]},
    )
    if already_committed:
        result = service.commit_completed_voice_pair(
            destination=destination, context=context
        )
        assert result.already_committed
    else:
        with pytest.raises(RuntimeError):
            service.commit_completed_voice_pair(
                destination=destination, context=context
            )
        assert _promotion_identity_rows(db_instance) == ([], [])


def test_commit_completed_voice_pair_atomically_persists_terminal_pair(db_instance):
    conversation_id, _root_id, destination, context = _voice_promotion_case(db_instance)
    service = ChatPersistenceService(db_instance)
    identities = derive_voice_promotion_identities(context.promotion_id)

    committed = service.commit_completed_voice_pair(
        destination=destination,
        context=context,
    )

    assert committed.conversation_id == conversation_id
    assert committed.user_message_id == identities.user_message_id
    assert committed.assistant_message_id == identities.assistant_message_id
    assert committed.terminal_receipt_id == identities.terminal_receipt_id
    assert committed.active_leaf_message_id == identities.assistant_message_id
    assert committed.already_committed is False
    revision_rows = (
        db_instance.get_connection()
        .execute(
            """SELECT source_message_id, revision_id
             FROM console_trace_semantic_revisions
            WHERE source_message_id IN (?, ?)
              AND revision_sequence = 0""",
            (identities.user_message_id, identities.assistant_message_id),
        )
        .fetchall()
    )
    revisions_by_message = {
        row["source_message_id"]: row["revision_id"] for row in revision_rows
    }
    assert (
        committed.user_revision_id == revisions_by_message[identities.user_message_id]
    )
    assert (
        committed.assistant_revision_id
        == revisions_by_message[identities.assistant_message_id]
    )
    user = db_instance.get_message_by_id(identities.user_message_id)
    assistant = db_instance.get_message_by_id(identities.assistant_message_id)
    assert user is not None
    assert user["conversation_id"] == conversation_id
    assert user["sender"] == "user"
    assert user["role"] == "user"
    assert user["content"] == context.user_text
    assert user["parent_message_id"] == destination.expected_persisted_leaf_id
    assert user["usage_json"] is None
    assert assistant is not None
    assert assistant["conversation_id"] == conversation_id
    assert assistant["sender"] == "assistant"
    assert assistant["role"] == "assistant"
    assert assistant["content"] == context.assistant_text
    assert assistant["parent_message_id"] == identities.user_message_id
    assert assistant["usage_json"] == context.usage_json
    assert (
        assistant["metadata_json"]
        == MessageMetadata(terminal_receipt_id=identities.terminal_receipt_id).to_json()
    )
    assert assistant["assistant_generation_state"] == "complete"
    marks = ConversationLocalMarksService(db_instance)
    assert marks.list_console_unseen_marks() == (
        (conversation_id, identities.terminal_receipt_id),
    )
    assert (
        marks.console_terminal_outcome(conversation_id, identities.terminal_receipt_id)
        == "complete"
    )
    assert db_instance.get_conversation_active_leaf(conversation_id) == (
        identities.assistant_message_id
    )


def test_commit_completed_voice_pair_accepts_same_incarnation_first_persistence(
    db_instance,
):
    conversation_id, root_id, destination, original_context = _voice_promotion_case(
        db_instance
    )
    temporary_origin = ConsoleSessionBindingOrigin(
        session_id=original_context.origin.session_id,
        session_incarnation=original_context.origin.session_incarnation,
        persisted_conversation_id=None,
        conversation_binding_revision=original_context.origin.conversation_binding_revision,
    )
    context = VoicePromotionContext(
        promotion_id=original_context.promotion_id,
        attempt_id=original_context.attempt_id,
        origin=temporary_origin,
        expected_native_leaf_id=original_context.expected_native_leaf_id,
        expected_persisted_leaf_id=None,
        user_text=original_context.user_text,
        assistant_text=original_context.assistant_text,
        usage_json=original_context.usage_json,
        terminal_boundary_id=original_context.terminal_boundary_id,
        capture_eligible_at_dispatch=original_context.capture_eligible_at_dispatch,
    )

    committed = ChatPersistenceService(db_instance).commit_completed_voice_pair(
        destination=destination,
        context=context,
    )

    user = db_instance.get_message_by_id(committed.user_message_id)
    assert committed.conversation_id == conversation_id
    assert user is not None
    assert user["parent_message_id"] == root_id


def test_commit_completed_voice_pair_active_leaf_conflict_creates_no_facts(
    db_instance,
):
    conversation_id, root_id, destination, context = _voice_promotion_case(db_instance)
    competing_id = db_instance.add_message(
        {
            "conversation_id": conversation_id,
            "parent_message_id": root_id,
            "sender": "user",
            "content": "competing durable turn",
            "id": "voice-competing-leaf",
        }
    )
    db_instance.set_conversation_active_leaf(conversation_id, competing_id)

    with pytest.raises(RuntimeError, match="conflict"):
        ChatPersistenceService(db_instance).commit_completed_voice_pair(
            destination=destination,
            context=context,
        )

    messages, marks = _promotion_identity_rows(db_instance)
    assert messages == []
    assert marks == []
    assert db_instance.get_conversation_active_leaf(conversation_id) == competing_id


def test_commit_completed_voice_pair_rejects_managed_ambient_transaction(
    db_instance,
):
    conversation_id, root_id, destination, context = _voice_promotion_case(db_instance)

    with db_instance.transaction(immediate=True):
        with pytest.raises(RuntimeError, match="own transaction"):
            ChatPersistenceService(db_instance).commit_completed_voice_pair(
                destination=destination,
                context=context,
            )

    assert _promotion_identity_rows(db_instance) == ([], [])
    assert db_instance.get_conversation_active_leaf(conversation_id) == root_id


def test_commit_completed_voice_pair_rejects_native_ambient_transaction(db_instance):
    conversation_id, root_id, destination, context = _voice_promotion_case(db_instance)
    connection = db_instance.get_connection()
    connection.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(RuntimeError, match="own transaction"):
            ChatPersistenceService(db_instance).commit_completed_voice_pair(
                destination=destination,
                context=context,
            )
    finally:
        connection.rollback()

    assert _promotion_identity_rows(db_instance) == ([], [])
    assert db_instance.get_conversation_active_leaf(conversation_id) == root_id


@pytest.mark.parametrize(
    "failure_boundary",
    ("user", "assistant", "unseen", "outcome", "active_leaf"),
)
def test_commit_completed_voice_pair_rolls_back_after_each_write_boundary(
    db_instance,
    monkeypatch,
    failure_boundary,
):
    conversation_id, root_id, destination, context = _voice_promotion_case(db_instance)
    service = ChatPersistenceService(db_instance)
    identities = derive_voice_promotion_identities(context.promotion_id)

    if failure_boundary in {"user", "assistant"}:
        original_add_message = db_instance.add_message

        def add_message_then_fail(payload):
            result = original_add_message(payload)
            expected = (
                identities.user_message_id
                if failure_boundary == "user"
                else identities.assistant_message_id
            )
            if payload.get("id") == expected:
                raise RuntimeError(f"after {failure_boundary} write")
            return result

        monkeypatch.setattr(db_instance, "add_message", add_message_then_fail)
    elif failure_boundary in {"unseen", "outcome"}:
        original_set_mark = service.local_marks.set_mark_with_cursor
        call_count = 0

        def set_mark_then_fail(*args, **kwargs):
            nonlocal call_count
            original_set_mark(*args, **kwargs)
            call_count += 1
            expected_count = 1 if failure_boundary == "unseen" else 2
            if call_count == expected_count:
                raise RuntimeError(f"after {failure_boundary} write")

        monkeypatch.setattr(
            service.local_marks,
            "set_mark_with_cursor",
            set_mark_then_fail,
        )
    else:
        original_cas = service._compare_and_swap_completed_voice_pair_leaf

        def cas_then_fail(*args, **kwargs):
            original_cas(*args, **kwargs)
            raise RuntimeError("after active_leaf write")

        monkeypatch.setattr(
            service,
            "_compare_and_swap_completed_voice_pair_leaf",
            cas_then_fail,
        )

    with pytest.raises(RuntimeError, match=f"after {failure_boundary} write"):
        service.commit_completed_voice_pair(destination=destination, context=context)

    messages, marks = _promotion_identity_rows(db_instance)
    assert messages == []
    assert marks == []
    assert (
        db_instance.get_connection()
        .execute(
            """SELECT count(*) FROM console_trace_semantic_revisions
             WHERE live_message_id IN (?, ?)""",
            (identities.user_message_id, identities.assistant_message_id),
        )
        .fetchone()[0]
        == 0
    )
    assert db_instance.get_conversation_active_leaf(conversation_id) == root_id


def test_commit_completed_voice_pair_reconciles_after_reported_post_commit_failure(
    db_instance,
    monkeypatch,
):
    conversation_id, _root_id, destination, context = _voice_promotion_case(db_instance)
    service = ChatPersistenceService(db_instance)
    original_transaction = db_instance.transaction
    depth = 0
    raised_after_commit = False

    @contextmanager
    def transaction_then_report_failure(*, immediate=False):
        nonlocal depth, raised_after_commit
        outermost = depth == 0
        depth += 1
        committed = False
        try:
            with original_transaction(immediate=immediate) as cursor:
                yield cursor
            committed = True
        finally:
            depth -= 1
        if outermost and committed and not raised_after_commit:
            raised_after_commit = True
            raise RuntimeError("reported after commit")

    monkeypatch.setattr(db_instance, "transaction", transaction_then_report_failure)
    with pytest.raises(RuntimeError, match="reported after commit"):
        service.commit_completed_voice_pair(destination=destination, context=context)
    monkeypatch.setattr(db_instance, "transaction", original_transaction)

    reconciled = service.commit_completed_voice_pair(
        destination=destination,
        context=context,
    )

    identities = derive_voice_promotion_identities(context.promotion_id)
    assert reconciled.already_committed is True
    assert reconciled.active_leaf_message_id == identities.assistant_message_id
    messages, marks = _promotion_identity_rows(db_instance)
    assert len(messages) == 2
    assert len(marks) == 2
    assert db_instance.get_conversation_active_leaf(conversation_id) == (
        identities.assistant_message_id
    )


def _insert_raw_voice_promotion_identity_set(
    db: CharactersRAGDB,
    *,
    conversation_id: str,
    root_id: str,
    context: VoicePromotionContext,
    mutation: str,
) -> None:
    identities = derive_voice_promotion_identities(context.promotion_id)
    now = db._get_current_utc_timestamp_iso()
    message_sql = """INSERT INTO messages(
          id, conversation_id, parent_message_id, sender, content,
          image_data, image_mime_type, timestamp, ranking, last_modified,
          deleted, client_id, version, feedback, role, variant_of,
          variant_number, is_selected_variant, total_variants, usage_json,
          metadata_json, provider_continuation_json, assistant_generation_state,
          thinking_blocks_json
        ) VALUES (?, ?, ?, ?, ?, NULL, NULL, ?, NULL, ?, 0, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL)"""
    user_feedback = "1;voice-user" if mutation == "user_feedback" else None
    assistant_feedback = (
        "1;voice-assistant" if mutation == "assistant_feedback" else None
    )
    variant_of = root_id if mutation == "variant_state" else None
    variant_number = 2 if mutation == "variant_state" else 1
    is_selected_variant = 0 if mutation == "variant_state" else 1
    total_variants = 2 if mutation == "variant_state" else 1
    client_id = "wrong-client" if mutation == "client_id" else db.client_id
    user_last_modified = "2000-01-01T00:00:00Z" if mutation == "timestamps" else now
    revision_created_at = {
        "revision_timestamp_blank": "",
        "revision_timestamp_naive": "2026-08-30T12:00:00.000",
        "revision_timestamp_non_utc": "2026-08-30T12:00:00.000+01:00",
        "revision_timestamp_garbage": "not-a-timestamp",
    }.get(mutation, now)
    with db.transaction(immediate=True) as cursor:
        cursor.execute(
            message_sql,
            (
                identities.user_message_id,
                conversation_id,
                root_id,
                "user",
                context.user_text,
                now,
                user_last_modified,
                client_id,
                user_feedback,
                "user",
                variant_of,
                variant_number,
                is_selected_variant,
                total_variants,
                None,
                None,
                None,
            ),
        )
        cursor.execute(
            message_sql,
            (
                identities.assistant_message_id,
                conversation_id,
                identities.user_message_id,
                "assistant",
                context.assistant_text,
                now,
                now,
                db.client_id,
                assistant_feedback,
                "assistant",
                None,
                1,
                1,
                1,
                context.usage_json,
                MessageMetadata(
                    terminal_receipt_id=identities.terminal_receipt_id
                ).to_json(),
                "complete",
            ),
        )
        if mutation == "missing_semantic_revisions":
            return
        revision_ids: dict[str, str] = {}
        revision_sql = """INSERT INTO console_trace_semantic_revisions(
              revision_id, source_conversation_id, source_message_id,
              revision_sequence, normalized_role, content_kind,
              creation_reason, predecessor_revision_id, live_message_id,
              live_locator_retired_at, created_at
            ) VALUES (?, ?, ?, 0, ?, 'text', 'message_create', NULL, ?, NULL, ?)"""
        for message_id, role in (
            (identities.user_message_id, "user"),
            (identities.assistant_message_id, "assistant"),
        ):
            revision_id = str(uuid4())
            revision_ids[message_id] = revision_id
            cursor.execute(
                revision_sql,
                (
                    revision_id,
                    conversation_id,
                    message_id,
                    (
                        "assistant"
                        if mutation == "mismatched_semantic_revision"
                        and message_id == identities.user_message_id
                        else role
                    ),
                    message_id,
                    revision_created_at,
                ),
            )
        if mutation == "extra_semantic_revision":
            cursor.execute(
                """INSERT INTO console_trace_semantic_revisions(
                      revision_id, source_conversation_id, source_message_id,
                      revision_sequence, normalized_role, content_kind,
                      creation_reason, predecessor_revision_id, live_message_id,
                      live_locator_retired_at
                    ) VALUES (?, ?, ?, 1, 'user', 'text', 'message_create',
                              ?, NULL, ?)""",
                (
                    str(uuid4()),
                    conversation_id,
                    identities.user_message_id,
                    revision_ids[identities.user_message_id],
                    now,
                ),
            )


def _seed_existing_voice_promotion_identity_set(
    db: CharactersRAGDB,
    *,
    conversation_id: str,
    root_id: str,
    context: VoicePromotionContext,
    mutation: str,
) -> None:
    identities = derive_voice_promotion_identities(context.promotion_id)
    other_conversation_id = db.add_conversation(
        {"title": "Wrong voice target", "character_id": None}
    )
    raw_mutations = {
        "missing_semantic_revisions",
        "extra_semantic_revision",
        "mismatched_semantic_revision",
        "user_feedback",
        "assistant_feedback",
        "variant_state",
        "client_id",
        "timestamps",
        "revision_timestamp_blank",
        "revision_timestamp_naive",
        "revision_timestamp_non_utc",
        "revision_timestamp_garbage",
    }
    if mutation in raw_mutations:
        _insert_raw_voice_promotion_identity_set(
            db,
            conversation_id=conversation_id,
            root_id=root_id,
            context=context,
            mutation=mutation,
        )
    else:
        user_conversation_id = (
            other_conversation_id if mutation == "conversation" else conversation_id
        )
        user_parent_id = None if mutation == "user_parent" else root_id
        user_role = "assistant" if mutation == "role" else "user"
        user_text = "wrong user text" if mutation == "user_text" else context.user_text
        user_payload = {
            "id": identities.user_message_id,
            "conversation_id": user_conversation_id,
            "parent_message_id": user_parent_id,
            "sender": "user",
            "role": user_role,
            "content": user_text,
        }
        if mutation == "attachment":
            user_message_id = db.add_message_with_semantic_sidecars(
                user_payload,
                attachments=(
                    {
                        "position": 1,
                        "data": b"voice-sidecar",
                        "mime_type": "text/plain",
                        "display_name": "voice.txt",
                    },
                ),
            )
        else:
            user_message_id = db.add_message(user_payload)
        assert user_message_id == identities.user_message_id
        assistant_parent_id = (
            root_id if mutation == "assistant_parent" else identities.user_message_id
        )
        assistant_text = (
            "wrong assistant text"
            if mutation == "assistant_text"
            else context.assistant_text
        )
        assistant_usage = (
            '{"output_tokens":999}' if mutation == "usage" else context.usage_json
        )
        metadata_receipt = (
            _OTHER_TERMINAL_RECEIPT
            if mutation == "terminal_metadata"
            else identities.terminal_receipt_id
        )
        assistant_state = "failed" if mutation == "terminal_state" else "complete"
        assistant_payload = {
            "id": identities.assistant_message_id,
            "conversation_id": conversation_id,
            "parent_message_id": assistant_parent_id,
            "sender": "assistant",
            "role": "assistant",
            "content": assistant_text,
            "usage_json": assistant_usage,
            "metadata_json": MessageMetadata(
                terminal_receipt_id=metadata_receipt
            ).to_json(),
            "assistant_generation_state": assistant_state,
        }
        if mutation == "generation_metadata":
            assistant_message_id = db.add_message_with_semantic_sidecars(
                assistant_payload,
                generation_metadata=(
                    {
                        "position": 0,
                        "prompt": "voice image",
                        "backend": "test",
                    },
                ),
            )
        else:
            assistant_message_id = db.add_message(assistant_payload)
        assert assistant_message_id == identities.assistant_message_id
        if mutation == "transcript_annotation":
            db.upsert_transcript_annotation(
                conversation_id=conversation_id,
                row_key=f"message:{identities.user_message_id}",
                message_id=identities.user_message_id,
                quote_text="exact private annotation quote",
                comment="exact private annotation comment",
            )
    marks = ConversationLocalMarksService(db)
    mark_conversation_id = (
        other_conversation_id if mutation == "unseen_conversation" else conversation_id
    )
    now = db._get_current_utc_timestamp_iso()
    mark_created_at = {
        "mark_timestamp_blank": "",
        "mark_timestamp_naive": "2026-08-30T12:00:00.000",
        "mark_timestamp_non_utc": "2026-08-30T12:00:00.000+01:00",
        "mark_timestamp_garbage": "not-a-timestamp",
    }.get(mutation, now)
    mark_updated_at = (
        "2026-08-30T12:00:01.000Z"
        if mutation == "mark_timestamp_diverged"
        else mark_created_at
    )
    with db.transaction(immediate=True) as cursor:
        if mutation == "missing_unseen":
            marks.set_mark_with_cursor(
                cursor,
                mark_conversation_id,
                marks.console_terminal_outcome_mark_type(
                    identities.terminal_receipt_id, "complete"
                ),
                created_at=now,
                updated_at=now,
            )
        else:
            marks.set_console_terminal_with_cursor(
                cursor,
                mark_conversation_id,
                identities.terminal_receipt_id,
                "failed" if mutation == "terminal_outcome" else "complete",
                created_at=now,
                updated_at=now,
            )
        if mutation.startswith("mark_timestamp_"):
            if mutation == "mark_timestamp_pair_mismatch":
                cursor.execute(
                    """UPDATE conversation_local_marks
                          SET created_at = ?, updated_at = ?
                        WHERE mark_type = ?""",
                    (
                        "2001-02-03T04:05:06.789Z",
                        "2001-02-03T04:05:06.789Z",
                        marks.console_terminal_outcome_mark_type(
                            identities.terminal_receipt_id, "complete"
                        ),
                    ),
                )
            else:
                cursor.execute(
                    """UPDATE conversation_local_marks
                          SET created_at = ?, updated_at = ?
                        WHERE mark_type IN (?, ?)""",
                    (
                        mark_created_at,
                        mark_updated_at,
                        marks.console_unseen_mark_type(identities.terminal_receipt_id),
                        marks.console_terminal_outcome_mark_type(
                            identities.terminal_receipt_id, "complete"
                        ),
                    ),
                )
    active_leaf_id = (
        root_id if mutation == "active_leaf" else identities.assistant_message_id
    )
    db.set_conversation_active_leaf(conversation_id, active_leaf_id)


@pytest.mark.parametrize(
    "mutation",
    (
        "conversation",
        "role",
        "user_text",
        "user_parent",
        "assistant_text",
        "assistant_parent",
        "usage",
        "terminal_metadata",
        "terminal_state",
        "missing_unseen",
        "terminal_outcome",
        "unseen_conversation",
        "active_leaf",
        "missing_semantic_revisions",
        "extra_semantic_revision",
        "mismatched_semantic_revision",
        "user_feedback",
        "assistant_feedback",
        "variant_state",
        "client_id",
        "timestamps",
        "revision_timestamp_blank",
        "revision_timestamp_naive",
        "revision_timestamp_non_utc",
        "revision_timestamp_garbage",
        "mark_timestamp_blank",
        "mark_timestamp_naive",
        "mark_timestamp_non_utc",
        "mark_timestamp_garbage",
        "mark_timestamp_diverged",
        "mark_timestamp_pair_mismatch",
        "attachment",
        "generation_metadata",
        "transcript_annotation",
    ),
)
def test_commit_completed_voice_pair_fails_closed_on_exact_identity_mismatch(
    db_instance,
    mutation,
):
    conversation_id, root_id, destination, context = _voice_promotion_case(db_instance)
    _seed_existing_voice_promotion_identity_set(
        db_instance,
        conversation_id=conversation_id,
        root_id=root_id,
        context=context,
        mutation=mutation,
    )
    before = _promotion_persistence_snapshot(
        db_instance,
        conversation_id=conversation_id,
        promotion_id=context.promotion_id,
    )
    if mutation == "mark_timestamp_pair_mismatch":
        terminal_marks = before["terminal_marks"]
        assert isinstance(terminal_marks, tuple)
        timestamp_pairs = {(row[2], row[3]) for row in terminal_marks}
        assert all(
            created_at == updated_at for created_at, updated_at in timestamp_pairs
        )
        assert len(timestamp_pairs) == 2

    with pytest.raises(RuntimeError, match="conflict"):
        ChatPersistenceService(db_instance).commit_completed_voice_pair(
            destination=destination,
            context=context,
        )

    assert (
        _promotion_persistence_snapshot(
            db_instance,
            conversation_id=conversation_id,
            promotion_id=context.promotion_id,
        )
        == before
    )


def test_commit_completed_voice_pair_fails_closed_on_partial_identity_set(db_instance):
    conversation_id, root_id, destination, context = _voice_promotion_case(db_instance)
    identities = derive_voice_promotion_identities(context.promotion_id)
    assert db_instance.add_message(
        {
            "id": identities.user_message_id,
            "conversation_id": conversation_id,
            "parent_message_id": root_id,
            "sender": "user",
            "content": context.user_text,
        }
    )
    before = _promotion_persistence_snapshot(
        db_instance,
        conversation_id=conversation_id,
        promotion_id=context.promotion_id,
    )

    with pytest.raises(RuntimeError, match="conflict"):
        ChatPersistenceService(db_instance).commit_completed_voice_pair(
            destination=destination,
            context=context,
        )

    assert (
        _promotion_persistence_snapshot(
            db_instance,
            conversation_id=conversation_id,
            promotion_id=context.promotion_id,
        )
        == before
    )


def test_commit_completed_voice_pair_retries_never_duplicate_rows_or_marks(db_instance):
    _conversation_id, _root_id, destination, context = _voice_promotion_case(
        db_instance
    )
    service = ChatPersistenceService(db_instance)

    first = service.commit_completed_voice_pair(
        destination=destination, context=context
    )
    second = service.commit_completed_voice_pair(
        destination=destination, context=context
    )
    third = service.commit_completed_voice_pair(
        destination=destination, context=context
    )

    assert first.already_committed is False
    assert second.already_committed is True
    assert third.already_committed is True
    messages, marks = _promotion_identity_rows(db_instance)
    assert len(messages) == 2
    assert len(marks) == 2


@pytest.mark.parametrize(
    "mutation",
    ("temporary_destination", "session", "incarnation", "capture_eligibility"),
)
def test_commit_completed_voice_pair_boundary_validation_precedes_writes(
    db_instance,
    mutation,
):
    conversation_id, root_id, destination, context = _voice_promotion_case(db_instance)
    values = {
        "session_id": destination.session_id,
        "session_incarnation": destination.session_incarnation,
        "persisted_conversation_id": destination.persisted_conversation_id,
        "expected_persisted_leaf_id": destination.expected_persisted_leaf_id,
        "capture_eligible_at_dispatch": destination.capture_eligible_at_dispatch,
    }
    if mutation == "temporary_destination":
        values["persisted_conversation_id"] = None
    elif mutation == "session":
        values["session_id"] = "different-session"
    elif mutation == "incarnation":
        values["session_incarnation"] = 2
    else:
        values["capture_eligible_at_dispatch"] = False
    invalid_destination = ResolvedVoicePromotionDestination(**values)

    with pytest.raises((TypeError, ValueError), match="destination"):
        ChatPersistenceService(db_instance).commit_completed_voice_pair(
            destination=invalid_destination,
            context=context,
        )

    assert _promotion_identity_rows(db_instance) == ([], [])
    assert db_instance.get_conversation_active_leaf(conversation_id) == root_id


def test_commit_completed_voice_pair_never_writes_workspace_database(
    db_instance,
    tmp_path,
):
    workspace_db = WorkspaceDB(
        tmp_path / "voice-workspaces.sqlite",
        client_id="voice-workspace-client",
    )
    registry = LocalWorkspaceRegistryService(workspace_db)
    registry.create_workspace(workspace_id="voice-workspace", name="Voice workspace")
    service = ChatPersistenceService(db_instance, workspace_registry=registry)
    _conversation_id, _root_id, destination, context = _voice_promotion_case(
        db_instance
    )
    try:
        with workspace_db.connection() as connection:
            changes_before = connection.total_changes

        service.commit_completed_voice_pair(destination=destination, context=context)

        with workspace_db.connection() as connection:
            assert connection.total_changes == changes_before
            assert (
                connection.execute(
                    "SELECT count(*) FROM workspace_memberships"
                ).fetchone()[0]
                == 0
            )
    finally:
        workspace_db.close()


def test_commit_completed_voice_pair_creates_no_provider_exchange_trace_lineage(
    db_instance,
):
    _conversation_id, _root_id, destination, context = _voice_promotion_case(
        db_instance
    )
    connection = db_instance.get_connection()
    provider_trace_tables = (
        "console_trace_segments",
        "console_trace_policies",
        "console_trace_artifacts",
        "console_trace_revision_bindings",
        "console_trace_surface_nodes",
        "console_trace_surface_replacements",
        "console_trace_request_headers",
        "console_trace_header_components",
        "console_trace_owners",
        "console_trace_calls",
        "console_trace_events",
        "console_trace_response_links",
        "console_trace_redaction_spans",
    )
    counts_before = {
        table: connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        for table in provider_trace_tables
    }

    ChatPersistenceService(db_instance).commit_completed_voice_pair(
        destination=destination,
        context=context,
    )

    assert {
        table: connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        for table in provider_trace_tables
    } == counts_before
    identities = derive_voice_promotion_identities(context.promotion_id)
    assert (
        connection.execute(
            """SELECT count(*) FROM console_trace_semantic_revisions
             WHERE live_message_id IN (?, ?)""",
            (identities.user_message_id, identities.assistant_message_id),
        ).fetchone()[0]
        == 2
    )
