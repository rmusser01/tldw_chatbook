"""Real transaction, scope and cleanup behavior of the local rule store."""

import sqlite3
from dataclasses import replace

import pytest

from Tests.Chat.response_rules_store_fixtures import activate, learning
from Tests.Chat.response_rules_store_fixtures import (
    rule_store as rule_store,  # noqa: PLC0414 - pytest fixture registration
)
from tldw_chatbook.Chat.response_rules.models import (
    RuleBinding,
    RuleLearningResult,
    RuleScope,
)
from tldw_chatbook.Chat.response_rules.repository import RuleBindingConflict

GLOBAL = RuleScope("global", "profile")


def effective(store, chat):
    return store.effective_rules(chat, None, GLOBAL)


def test_activation_write_failure_stays_inactive(rule_store):
    store, db, origin = rule_store
    scope = RuleScope("chat", origin.conversation_id)
    with db.transaction() as cursor:
        cursor.execute(
            "CREATE TEMP TRIGGER reject_rule_binding BEFORE INSERT ON console_response_rule_bindings BEGIN SELECT RAISE(ABORT, 'injected failure'); END"
        )
    with pytest.raises(sqlite3.DatabaseError):
        activate(store, origin)
    assert effective(store, scope) == ()
    with db.transaction() as cursor:
        assert (
            cursor.execute(
                "SELECT COUNT(*) FROM console_response_rule_revisions"
            ).fetchone()[0]
            == 0
        )


def test_failed_draft_without_revision_is_inspectable_only_in_its_scope(rule_store):
    store, _, origin = rule_store
    scope = RuleScope("chat", origin.conversation_id)
    failed = RuleLearningResult("inactive", None, None, {}, "invalid_candidate")
    store.save_draft(scope, origin, failed, complaint="private complaint")
    assert store.list_drafts(scope) == (failed,)
    assert store.list_drafts(GLOBAL) == ()
    assert effective(store, scope) == ()


def test_temporary_adoption_rolls_back_without_losing_live_rules(rule_store):
    store, db, durable = rule_store
    temporary = replace(durable, conversation_id=None, message_id="native-answer")
    scope = RuleScope("chat", temporary.session_id)
    activate(store, temporary, scope)
    before = effective(store, scope)
    with (
        pytest.raises(RuntimeError, match="rollback"),
        db.transaction(immediate=True) as cursor,
    ):
        store.adopt_temporary(
            temporary.session_id,
            durable.conversation_id,
            {temporary.message_id: durable.message_id},
            cursor,
        )
        assert effective(store, scope) == before
        raise RuntimeError("rollback")
    assert effective(store, scope) == before
    assert store.list_bindings(RuleScope("chat", durable.conversation_id)) == ()
    with db.transaction(immediate=True) as cursor:
        store.adopt_temporary(
            temporary.session_id,
            durable.conversation_id,
            {temporary.message_id: durable.message_id},
            cursor,
        )
    adopted = effective(store, RuleScope("chat", durable.conversation_id))
    assert len(adopted) == 1
    assert adopted[0].origin.conversation_id == durable.conversation_id
    assert adopted[0].origin.message_id == durable.message_id
    assert effective(store, scope) == ()


def test_fork_omits_chat_bindings_but_keeps_inherited_rules(rule_store):
    store, _, origin = rule_store
    activate(store, origin)
    assert effective(store, RuleScope("chat", "separate-fork")) == ()
    store.promote("rule", 1, GLOBAL, expected_binding_revision=0)
    assert [
        r.rule_id for r in effective(store, RuleScope("chat", "separate-fork"))
    ] == ["rule"]


def test_binding_compare_and_swap_and_prewrite_invalidation(rule_store):
    store, db, origin = rule_store
    binding = activate(store, origin)
    seen = []
    unsubscribe = store.add_invalidation_listener(
        lambda scope, rule_id: seen.append((scope, rule_id))
    )
    with db.transaction() as cursor:
        cursor.execute(
            "CREATE TEMP TRIGGER reject_rule_edit BEFORE UPDATE ON console_response_rule_bindings BEGIN SELECT RAISE(ABORT, 'injected failure'); END"
        )
    with pytest.raises(sqlite3.DatabaseError):
        store.set_binding(
            replace(binding, state="disabled"), expected_binding_revision=1
        )
    assert seen == [(binding.scope, "rule")]
    assert store.list_bindings(binding.scope)[0].state == "enabled"
    with pytest.raises(RuleBindingConflict):
        store.set_binding(
            replace(binding, state="disabled"), expected_binding_revision=0
        )
    unsubscribe()


def test_source_deletion_removes_fixture_bodies_not_promoted_definition(rule_store):
    store, db, origin = rule_store
    scope = RuleScope("chat", origin.conversation_id)
    result = learning(origin)
    store.save_draft(scope, origin, result, complaint="complaint")
    activate(store, origin)
    store.promote("rule", 1, GLOBAL, expected_binding_revision=0)
    before = store.get_revision("rule", 1)
    with db.transaction() as cursor:
        store.remove_source(
            origin.conversation_id, origin.message_id, permanent=False, cursor=cursor
        )
    assert store.list_drafts(scope)
    with db.transaction() as cursor:
        store.remove_source(
            origin.conversation_id, origin.message_id, permanent=True, cursor=cursor
        )
    assert store.list_drafts(scope) == ()
    assert store.get_validation("rule", 1) is None
    assert store.get_revision("rule", 1) == before
    assert effective(store, GLOBAL)[0] == before
    with db.transaction() as cursor:
        assert (
            cursor.execute(
                "SELECT COUNT(*) FROM console_response_rule_fixtures"
            ).fetchone()[0]
            == 0
        )


def test_recorded_case_is_a_version_reference_not_a_body_copy(rule_store):
    store, db, origin = rule_store
    result = learning(origin)
    store.save_draft(
        RuleScope("chat", origin.conversation_id),
        origin,
        result,
        complaint="private complaint",
    )
    with db.transaction() as cursor:
        row = cursor.execute(
            "SELECT message_id,message_version,input_json FROM console_response_rule_fixtures WHERE case_type='recorded_violation'"
        ).fetchone()
        assert (row[0], row[1], row[2]) == (
            origin.message_id,
            origin.message_version,
            None,
        )
        triggers = [
            r[0]
            for r in cursor.execute(
                "SELECT sql FROM sqlite_schema WHERE type='trigger' AND tbl_name LIKE 'console_response_rule_%'"
            )
        ]
        assert not any("sync_log" in sql or "_fts" in sql for sql in triggers)
    assert "private complaint" not in repr(result)


def test_delete_last_binding_cleans_definition_but_promoted_revision_survives(
    rule_store,
):
    store, _, origin = rule_store
    binding = activate(store, origin)
    store.promote("rule", 1, GLOBAL, expected_binding_revision=0)
    store.delete_binding(binding.scope, "rule", expected_binding_revision=1)
    assert store.get_revision("rule", 1).rule_id == "rule"
    store.delete_binding(GLOBAL, "rule", expected_binding_revision=1)
    with pytest.raises(KeyError):
        store.get_revision("rule", 1)


@pytest.mark.parametrize("fail_after_adoption", [False, True])
def test_actual_temporary_chat_save_adopts_rules_in_transcript_transaction(
    rule_store, fail_after_adoption
):
    from Tests.Chat.response_rules_fixtures import source
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    rules, db, _ = rule_store
    chats = ConsoleChatStore(
        persistence=ChatPersistenceService(db), response_rule_store=rules
    )
    session = chats.create_session(ephemeral=True)
    chats.append_message(
        session.id, role=ConsoleMessageRole.USER, content="Explain result"
    )
    answer = chats.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="Missing proof"
    )
    origin = source(session_id=session.id, conversation_id=None, message_id=answer.id)
    scope = RuleScope("chat", session.id)
    activate(rules, origin, scope)
    before = effective(rules, scope)

    class FailLater:
        def write(self, **kwargs):
            raise RuntimeError("later sidecar failed")

    if fail_after_adoption:
        with pytest.raises(RuntimeError, match="later sidecar"):
            chats.promote_ephemeral_session(session.id, contributions=(FailLater(),))
        assert session.ephemeral is True
        assert session.persisted_conversation_id is None
        assert effective(rules, scope) == before
        with db.transaction() as cursor:
            assert (
                cursor.execute(
                    "SELECT COUNT(*) FROM console_response_rule_bindings"
                ).fetchone()[0]
                == 0
            )
    else:
        conversation = chats.promote_ephemeral_session(session.id)
        assert session.ephemeral is False
        adopted = effective(rules, RuleScope("chat", conversation))
        assert len(adopted) == 1
        assert (
            adopted[0].origin.message_id
            == chats.get_message(answer.id).persisted_message_id
        )
        assert effective(rules, scope) == ()


def test_actual_temporary_close_discards_private_rule_memory(rule_store):
    from Tests.Chat.response_rules_fixtures import source
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    rules, db, _ = rule_store
    chats = ConsoleChatStore(
        persistence=ChatPersistenceService(db), response_rule_store=rules
    )
    session = chats.create_session(ephemeral=True)
    answer = chats.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="Missing proof"
    )
    origin = source(session_id=session.id, conversation_id=None, message_id=answer.id)
    scope = RuleScope("chat", session.id)
    activate(rules, origin, scope)
    rules.save_draft(scope, origin, learning(origin), complaint="private complaint")
    rules.promote("rule", 1, GLOBAL, expected_binding_revision=0)
    chats.close_session(session.id)
    assert rules.list_drafts(scope) == ()
    assert (
        effective(rules, scope)[0].rule_id == "rule"
    )  # Only explicit global promotion remains.
    assert rules.list_bindings(scope) == ()


def test_rule_writes_do_not_touch_chat_metadata_or_sync_log(rule_store):
    store, db, origin = rule_store
    with db.transaction() as cursor:
        before_log = cursor.execute("SELECT COUNT(*) FROM sync_log").fetchone()[0]
        before_metadata = cursor.execute(
            "SELECT metadata FROM conversations WHERE id=?", (origin.conversation_id,)
        ).fetchone()[0]
    store.save_draft(
        RuleScope("chat", origin.conversation_id),
        origin,
        learning(origin),
        complaint="private complaint",
    )
    activate(store, origin)
    with db.transaction() as cursor:
        assert (
            cursor.execute("SELECT COUNT(*) FROM sync_log").fetchone()[0] == before_log
        )
        assert (
            cursor.execute(
                "SELECT metadata FROM conversations WHERE id=?",
                (origin.conversation_id,),
            ).fetchone()[0]
            == before_metadata
        )


def test_untested_draft_cannot_be_enabled_or_promoted(rule_store):
    store, _, origin = rule_store
    scope = RuleScope("chat", origin.conversation_id)
    untested = replace(learning(origin), validation=None, fixtures={})
    store.save_draft(scope, origin, untested, complaint="edited criteria")
    with pytest.raises(ValueError, match="validation"):
        store.promote("rule", 1, GLOBAL, expected_binding_revision=0)
    with pytest.raises(ValueError, match="validation"):
        store.set_binding(
            RuleBinding(scope, "rule", 1, "enabled", 1), expected_binding_revision=0
        )
    assert effective(store, scope) == ()


def test_activation_revalidates_exact_source_version_in_transaction(rule_store):
    store, db, origin = rule_store
    changed = replace(origin, message_version=origin.message_version + 1)
    with pytest.raises(ValueError, match="stale_rule_source"):
        activate(store, changed)
    with db.transaction() as cursor:
        assert (
            cursor.execute(
                "SELECT COUNT(*) FROM console_response_rule_revisions"
            ).fetchone()[0]
            == 0
        )


@pytest.mark.parametrize("promoted", [False, True])
def test_permanent_chat_row_deletion_cleans_only_unreferenced_definitions(
    rule_store, promoted
):
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService

    store, db, origin = rule_store
    conversation = ChatPersistenceService(db).create_conversation(
        conversation_title="Rule owner"
    )
    scope = RuleScope("chat", conversation)
    activate(store, origin, scope)
    if promoted:
        store.promote("rule", 1, GLOBAL, expected_binding_revision=0)
    with db.transaction() as cursor:
        cursor.execute("DELETE FROM conversations WHERE id=?", (conversation,))
    assert store.list_bindings(scope) == ()
    if promoted:
        assert store.get_revision("rule", 1).rule_id == "rule"
    else:
        with pytest.raises(KeyError):
            store.get_revision("rule", 1)
