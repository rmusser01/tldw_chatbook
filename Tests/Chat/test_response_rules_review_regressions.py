"""Public lifecycle regressions from the independent whole-branch review."""

from dataclasses import replace

import pytest

from Tests.Chat.response_rules_fixtures import source
from Tests.Chat.response_rules_store_fixtures import learning, rule_store
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.response_rules.models import RuleScope
from tldw_chatbook.Chat.response_rules.repository import RuleBindingConflict


def temporary(rule_store):
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
    origin = source(
        session_id=session.id,
        conversation_id=None,
        message_id=answer.id,
        message_version=chats.response_rule_source_version(answer.id),
    )
    result = learning(origin)
    scope = RuleScope("chat", session.id)
    rules.save_draft(scope, origin, result, complaint="private complaint")
    rules.activate(result.rule, result.validation, scope, expected_binding_revision=0)
    return rules, chats, session, answer, scope


def test_deleted_temporary_example_does_not_prevent_save(rule_store):
    rules, chats, session, answer, scope = temporary(rule_store)
    chats.delete_message(answer.id)
    assert rules.list_drafts(scope) == ()
    conversation = chats.promote_ephemeral_session(session.id)
    global_scope = RuleScope("global", "profile")
    surviving = rules.effective_rules(
        RuleScope("chat", conversation), None, global_scope
    )
    assert [r.rule_id for r in surviving] == ["rule"]
    assert surviving[0].origin.conversation_id is None


def test_promoted_temporary_rule_can_be_reenabled_after_close(rule_store):
    rules, chats, session, _answer, _scope = temporary(rule_store)
    destination = RuleScope("global", "profile")
    binding = rules.promote("rule", 1, destination, expected_binding_revision=0)
    chats.close_session(session.id)
    disabled = rules.set_binding(
        replace(binding, state="disabled"), expected_binding_revision=1
    )
    rules.set_binding(replace(disabled, state="enabled"), expected_binding_revision=2)
    assert rules.list_bindings(destination)[0].state == "enabled"
    validation = rules.get_validation("rule", 1)
    assert validation is not None and all(
        c.check.reason == "example_tested" for c in validation.case_results
    )
    assert rules.list_drafts(destination) == ()


def named_learning(origin, name):
    result = learning(origin)
    return replace(
        result,
        rule=replace(result.rule, rule_id=name),
        validation=replace(
            result.validation,
            case_results=tuple(
                replace(c, check=replace(c.check, rule_id=name))
                for c in result.validation.case_results
            ),
        ),
    )


def fill(rules, origin, scope, count=16):
    for i in range(count):
        result = named_learning(origin, f"rule-{i}")
        rules.activate(
            result.rule, result.validation, scope, expected_binding_revision=0
        )


def test_seventeenth_activation_refuses_before_publication(rule_store):
    rules, _db, origin = rule_store
    scope = RuleScope("chat", origin.conversation_id)
    fill(rules, origin, scope)
    extra = named_learning(origin, "extra")
    with pytest.raises(ValueError, match="too_many_effective_rules"):
        rules.activate(extra.rule, extra.validation, scope, expected_binding_revision=0)
    assert len(rules.list_bindings(scope)) == 16


def test_enable_refuses_seventeenth_effective_rule(rule_store):
    rules, _db, origin = rule_store
    scope = RuleScope("chat", origin.conversation_id)
    fill(rules, origin, scope, 15)
    extra = named_learning(origin, "extra")
    binding = rules.activate(
        extra.rule, extra.validation, scope, expected_binding_revision=0
    )
    disabled = rules.set_binding(
        replace(binding, state="disabled"), expected_binding_revision=1
    )
    fill(rules, origin, RuleScope("global", "profile"), 1)
    # Use a distinct inherited identity, since logical overrides count once.
    inherited = named_learning(origin, "inherited")
    rules.activate(
        inherited.rule,
        inherited.validation,
        RuleScope("global", "profile"),
        expected_binding_revision=0,
    )
    with pytest.raises(ValueError, match="too_many_effective_rules"):
        rules.set_binding(
            replace(disabled, state="enabled"), expected_binding_revision=2
        )
    assert (
        next(b for b in rules.list_bindings(scope) if b.rule_id == "extra").state
        == "disabled"
    )


def test_global_promotion_refuses_overflow_in_an_existing_chat(rule_store):
    rules, db, origin = rule_store
    scope = RuleScope("chat", origin.conversation_id)
    fill(rules, origin, scope)
    other = ChatPersistenceService(db).create_conversation(conversation_title="Other")
    extra = named_learning(origin, "extra")
    rules.activate(
        extra.rule,
        extra.validation,
        RuleScope("chat", other),
        expected_binding_revision=0,
    )
    with pytest.raises(ValueError, match="too_many_effective_rules"):
        rules.promote(
            "extra", 1, RuleScope("global", "profile"), expected_binding_revision=0
        )
    assert rules.list_bindings(RuleScope("global", "profile")) == ()


@pytest.mark.parametrize("action", ["set", "delete", "promote"])
def test_profile_management_refuses_changed_owner_before_write(rule_store, action):
    rules, _db, origin = rule_store
    result = learning(origin)
    scope = RuleScope("chat", origin.conversation_id)
    binding = rules.activate(
        result.rule, result.validation, scope, expected_binding_revision=0
    )
    with pytest.raises(RuleBindingConflict, match="rule_scope_changed"):
        if action == "set":
            rules.set_binding(
                replace(binding, state="disabled"),
                expected_binding_revision=1,
                current=lambda: False,
            )
        elif action == "delete":
            rules.delete_binding(
                scope, "rule", expected_binding_revision=1, current=lambda: False
            )
        else:
            rules.promote(
                "rule",
                1,
                RuleScope("global", "profile"),
                expected_binding_revision=0,
                current=lambda: False,
            )
    assert rules.list_bindings(scope)[0].state == "enabled"
