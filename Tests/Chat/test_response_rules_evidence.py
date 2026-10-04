"""Capture from native branch/result owners without retrieving new content."""

from dataclasses import replace

import pytest

from Tests.Chat.response_rules_fixtures import source
from tldw_chatbook.Agents.agent_models import ToolResult
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.response_rules.evidence import capture_rule_input


def exchange():
    store = ConsoleChatStore()
    session = store.ensure_session()
    user = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="Check the original work"
    )
    answer = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="I have not verified it"
    )
    origin = source(
        session_id=session.id,
        conversation_id=None,
        branch_id=answer.id,
        message_id=answer.id,
        message_version=store.response_rule_source_version(answer.id),
    )
    return store, session, user, answer, origin


def test_capture_contains_only_original_user_text_and_response():
    store, _session, user, answer, origin = exchange()
    value = capture_rule_input(store, origin)
    assert value.request_text == user.content and value.response_text == answer.content
    assert value.evidence == () and not value.evidence_complete
    assert value.work_revision is None


@pytest.mark.parametrize(
    "change", ["branch", "version", "missing", "other_chat", "conversation"]
)
def test_capture_rejects_wrong_branch_or_changed_source(change):
    store, session, _user, answer, origin = exchange()
    if change == "branch":
        origin = replace(origin, branch_id="wrong")
    if change == "version":
        origin = replace(origin, message_version=origin.message_version + 1)
    if change == "missing":
        origin = replace(origin, message_id="missing")
    if change == "other_chat":
        origin = replace(origin, session_id=store.create_session().id)
    if change == "conversation":
        origin = replace(origin, conversation_id="wrong-owner")
    with pytest.raises(ValueError, match="rule_source"):
        capture_rule_input(store, origin)
    assert store.get_message(answer.id).content == answer.content
    assert store.active_leaf(session.id) == answer.id


@pytest.mark.parametrize("observed,size", [(True, 10), (False, 10), (True, 70000)])
def test_capture_pairs_provider_bound_body_with_host_dispatch_facts(observed, size):
    from tldw_chatbook.Chat.provider_continuation import (
        ContinuationCall,
        ContinuationResult,
        ContinuationRound,
        ProviderContinuationCheckpoint,
    )

    store, _session, _user, answer, origin = exchange()
    store._message_or_raise(answer.id).provider_continuation = (
        ProviderContinuationCheckpoint(
            2,
            1,
            "moonshot",
            "chat_completions",
            "model",
            "PRIVATE-ENDPOINT",
            "complete",
            (
                ContinuationRound(
                    "PRIVATE-OTHER-ANSWER",
                    ("PRIVATE-REASONING",),
                    (
                        ContinuationCall(
                            "call",
                            "fs_read",
                            "PRIVATE-ARGS",
                            "completed",
                            ContinuationResult("x" * size),
                        ),
                    ),
                ),
            ),
        )
    )
    if observed:
        store.record_response_rule_tool_result(
            answer.id, "call", state="settled", outcome="succeeded", tool_name="fs_read"
        )
    value = capture_rule_input(store, origin)
    assert value.evidence[0].text == "x" * min(size, 65536)
    assert value.evidence[0].state == ("settled" if observed else "uncertain")
    assert value.evidence_complete is (observed and size <= 65536)
    assert "PRIVATE-" not in repr(value)


def test_prior_chain_settled_results_are_retained_without_private_result_bodies():
    store, session, user, answer, _origin = exchange()
    store.record_response_rule_tool_result(
        answer.id,
        "old-test",
        state="settled",
        outcome="succeeded",
        tool_name="run_tests",
    )
    store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="Correct the answer"
    )
    correction = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Earlier checks succeeded, current freshness is unknown",
    )
    store.bind_response_rule_task_root(correction.id, user.id)
    origin = source(
        session_id=session.id,
        conversation_id=None,
        branch_id=correction.id,
        message_id=correction.id,
        message_version=store.response_rule_source_version(correction.id),
    )
    value = capture_rule_input(store, origin)
    assert value.evidence[0].state == "settled"
    assert value.evidence[0].outcome == "succeeded"
    assert value.evidence[0].work_revision is None
    assert not value.evidence_complete
    # Task-chain request selection is explicit; unrelated earlier exchanges are excluded.
    assert value.request_text == user.content
    assert user.content not in value.response_text


def test_inactive_branch_results_are_never_captured():
    store, session, user, answer, _origin = exchange()
    store.record_response_rule_tool_result(
        answer.id,
        "private-old-branch",
        state="settled",
        outcome="succeeded",
        tool_name="PRIVATE-OLD-TOOL",
    )
    store.set_active_leaf(session.id, user.id)
    sibling = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="Sibling answer"
    )
    origin = source(
        session_id=session.id,
        conversation_id=None,
        branch_id=sibling.id,
        message_id=sibling.id,
        message_version=store.response_rule_source_version(sibling.id),
    )
    assert capture_rule_input(store, origin).evidence == ()


def test_controller_callback_retains_actual_dispatch_state_without_raw_sensitive_body():
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    store, session, _user, answer, origin = exchange()
    controller = ConsoleChatController(store=store, provider_gateway=object())
    controller.observe_response_rule_tool_result(
        session.id,
        answer.id,
        "run",
        "call",
        "fs_read",
        ToolResult(
            ok=False, error="RAW-SENSITIVE-CANARY", dispatch_state="not_started"
        ),
    )
    value = capture_rule_input(store, origin)
    assert value.evidence[0].state == "not_started"
    assert value.evidence[0].outcome == "failed"
    assert "RAW-SENSITIVE-CANARY" not in value.evidence[0].text
    assert "RAW-SENSITIVE-CANARY" not in repr(value)


def test_explicit_scope_root_retains_original_request_for_correction_chain():
    store, session, user, _answer, _origin = exchange()
    feedback = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="PRIVATE-CORRECTION-GUIDANCE"
    )
    correction = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="Fixed answer"
    )
    store.bind_response_rule_task_root(correction.id, user.id)
    origin = source(
        session_id=session.id,
        conversation_id=None,
        branch_id=correction.id,
        message_id=correction.id,
        message_version=store.response_rule_source_version(correction.id),
    )
    value = capture_rule_input(store, origin)
    assert value.request_text == user.content
    assert feedback.content not in value.request_text


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
@pytest.mark.requires_cleanup
async def test_actual_agent_send_binds_definitive_callback_to_response_owner(
    tmp_path, monkeypatch
):
    from Tests.Chat.test_console_chat_controller import (
        test_controller_bridge_agent_service_bound_private_history_on_real_send,
    )

    await test_controller_bridge_agent_service_bound_private_history_on_real_send(
        tmp_path, monkeypatch
    )
