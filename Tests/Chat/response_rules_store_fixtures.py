"""Real private SQLite owners and independently labelled calibration inputs."""

import shutil
from dataclasses import replace

import pytest

from Tests.Chat.response_rules_fixtures import inputs, revision, source
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.response_rules.models import (
    RuleCaseResult,
    RuleCheck,
    RuleLearningResult,
    RuleScope,
    RuleValidation,
)
from tldw_chatbook.Chat.response_rules.repository import ResponseRuleRepository
from tldw_chatbook.Chat.response_rules.store import ResponseRuleStore
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def rule_store(tmp_path, chachanotes_template_db):
    path = tmp_path / "rules.sqlite"
    shutil.copyfile(chachanotes_template_db, path)
    db = CharactersRAGDB(path, "rules-tests")
    service = ChatPersistenceService(db)
    conversation = service.create_conversation(conversation_title="Rules")
    user = service.create_message(
        conversation_id=conversation, sender="user", content="Explain result"
    )
    answer = service.create_message(
        conversation_id=conversation,
        sender="assistant",
        content="Missing proof",
        parent_message_id=user,
    )
    try:
        yield ResponseRuleStore(ResponseRuleRepository(db)), db, source(
            conversation_id=conversation, message_id=answer
        )
    finally:
        db.close()


def learning(origin=None):
    origin = origin or source()
    rule = replace(revision(), origin=origin)
    kinds = (
        "recorded_violation",
        "synthetic_violation",
        "synthetic_correction",
        "synthetic_acceptable",
    )
    cases = tuple(
        RuleCaseResult(
            f"case-{i}",
            kind,
            RuleCheck(
                "rule", 1, "applicable", "violation" if i < 2 else "pass", "", ()
            ),
        )
        for i, kind in enumerate(kinds)
    )
    validation = RuleValidation(
        rule.candidate.candidate_digest(),
        rule.candidate.detector_digest(),
        1,
        "provider",
        "model",
        cases,
        origin,
        tuple(c.case_id for c in cases),
        "2026-10-04T00:00:00Z",
    )
    fixtures = {
        c.case_id: inputs("Missing proof" if i < 2 else "Evidence")
        for i, c in enumerate(cases)
    }
    return RuleLearningResult("inactive", rule, validation, fixtures, "tested")


def activate(store, origin, scope=None):
    learned = learning(origin)
    scope = scope or RuleScope("chat", origin.conversation_id or origin.session_id)
    return store.activate(
        learned.rule, learned.validation, scope, expected_binding_revision=0
    )
