"""Saved thinking response links require exact typed evidence and frozen meaning."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_trace_settlement import _call, _request
from Tests.Chat.test_console_trace_settlement import (
    db as db,  # noqa: PLC0414 - pytest fixture re-export
)
from Tests.Chat.test_console_trace_sidecar_replay import (
    replay_harness as replay_harness,  # noqa: PLC0414 - pytest fixture re-export
)
from tldw_chatbook.Chat.console_semantic_revision import (
    project_semantic_revision_trace_message,
)
from tldw_chatbook.Chat.console_trace_models import FrozenTracePolicy, new_opaque_id
from tldw_chatbook.Chat.console_trace_native_reader import ConsoleTraceNativeReader
from tldw_chatbook.Chat.console_trace_redaction import BUILTIN_PII_RULESET_REVISION_ID
from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
from tldw_chatbook.Chat.console_trace_runtime import ConsoleTraceBoundaryFactory
from tldw_chatbook.Chat.console_trace_settlement import (
    ConsoleTraceSettlementCoordinator,
)
from tldw_chatbook.Chat.thinking_blocks import (
    DisplayableThinkingBlock,
    ThinkingEnvelope,
    dump_thinking_blocks_json,
)

PROFILE = {
    "version": 1,
    "thinking_stream_disposition": "displayable",
    "thinking_round_trip_version": 1,
    "provider": "openai",
    "model": "gpt-test",
    "protocol": "chat_completions",
    "source_format": "start_anchored_think",
}


@pytest.mark.parametrize(
    "case",
    [
        "exact",
        "fragmented",
        "pii",
        "pii_mismatch",
        "pii_unavailable",
        "pii_legacy_mismatch",
        "pii_legacy_unavailable",
        "pii_legacy_exact",
        "pii_legacy_exact_unavailable",
        "queued",
        "raw_mismatch",
        "provider",
        "model",
        "format",
        "status",
        "round",
        "legacy",
        "ignored",
        "malformed",
    ],
)
def test_saved_thinking_response_projection_requires_exact_evidence(
    db, case, monkeypatch
):
    repository = ConsoleTraceRepository()
    coordinator = ConsoleTraceSettlementCoordinator(repository)
    profile = None if "legacy" in case else dict(PROFILE)
    if case == "ignored":
        profile["thinking_stream_disposition"] = "ignored"
    conversation_id, _segment, call_id = _call(
        db,
        repository,
        response_projection=profile,
        policy=(
            FrozenTracePolicy(
                new_opaque_id(), "credentials-v1", True, BUILTIN_PII_RULESET_REVISION_ID
            )
            if case.startswith("pii")
            else None
        ),
    )
    secret = (
        "trace.person@example.com"
        if case.startswith("pii")
        else "Bearer sk-proj-" + "x" * 48
    )
    block = DisplayableThinkingBlock(
        block_id="saved-response-thinking",
        round_ordinal=0,
        provider="openai",
        model="gpt-test",
        protocol="chat_completions",
        source_format="start_anchored_think",
        status="complete",
        text=secret,
    )
    if case in {"provider", "model"}:
        block = replace(block, **{case: "different"})
    elif case == "format":
        block = replace(block, source_format="reasoning_content")
    elif case == "status":
        block = replace(block, status="stopped")
    elif case == "round":
        block = replace(block, round_ordinal=1)
    elif case in {"pii_mismatch", "pii_legacy_mismatch"}:
        block = replace(block, text="other.person@example.com")
    assistant_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": secret if "legacy_exact" in case else "answer",
            "thinking_blocks_json": dump_thinking_blocks_json(
                ThinkingEnvelope((block,))
            ),
            "assistant_generation_state": "complete",
        }
    )
    assert assistant_id is not None
    thinking = {
        key: PROFILE[key] for key in ("provider", "model", "protocol", "source_format")
    }
    thinking["text"] = (
        "Bearer sk-proj-" + "y" * 48 if case == "raw_mismatch" else secret
    )
    response = {"role": "assistant", "content": "answer", "thinking": [thinking]}
    if "legacy_exact" in case:
        response = {"role": "assistant", "content": secret}
    if case == "fragmented":
        response["thinking"] = [
            dict(thinking, text=secret[:13]),
            dict(thinking, text=secret[13:]),
        ]
    elif case == "malformed":
        response["thinking"] = [dict(thinking, unexpected=True)]
    elif case in {"legacy", "ignored"}:
        # Existing visible-only references must never gain thinking on read.
        response.pop("thinking")
    request = _request(call_id, canonical_message_id=assistant_id, response=response)
    if case.endswith("unavailable"):
        monkeypatch.setattr(
            "tldw_chatbook.Chat.console_trace_settlement.redact_pii_value_for_ruleset_revision",
            lambda *_args: SimpleNamespace(
                available=False, omission_reason_code="pii_detector_unavailable"
            ),
        )
    handoff = coordinator.prepare_handoff(db, request)
    assert secret not in repr(handoff)
    if case == "queued":
        settle = coordinator._settle_prepared

        def unavailable(*_args):
            raise OSError("isolated busy database")

        monkeypatch.setattr(coordinator, "_settle_prepared", unavailable)
        assert not handoff.settle(assistant_id)
        assert coordinator.pending_count == 1
        pending = coordinator._pending[call_id][1]
        assert secret.encode() not in pending.response_bytes
        assert pending._response_equality_proof is not None
        assert pending._response_equality_proof.hex() not in "\n".join(
            db.get_connection().iterdump()
        )
        monkeypatch.setattr(coordinator, "_settle_prepared", settle)
        assert coordinator.retry_pending() == 1
    else:
        assert handoff.settle(assistant_id)
    cursor = db.get_connection().cursor()
    call = repository.get_call(cursor, call_id)
    link = repository.get_response_link(cursor, call_id)
    assert link is not None
    expects_revision = case in {
        "exact",
        "fragmented",
        "pii",
        "queued",
        "legacy",
        "pii_legacy_exact",
    }
    assert link.link_kind == ("revision" if expects_revision else "artifact")
    captured = ConsoleTraceNativeReader(db)._reconstruct_response(cursor, call)
    assert secret not in json.dumps(captured)
    if case == "pii":
        assert project_semantic_revision_trace_message(
            cursor,
            revision_id=link.semantic_revision_id,
            expected_conversation_id=conversation_id,
            policy_id=call.policy_id,
        ) == {"role": "assistant", "content": "answer"}
    if case.startswith("pii_"):
        assert coordinator.submit(db, request)
        assert (
            ConsoleTraceNativeReader(db)._reconstruct_response(cursor, call) == captured
        )
    if case in {"exact", "fragmented", "pii", "queued"}:
        assert captured["content"] == "answer" and len(captured["thinking"]) == 1
        assert captured["thinking"][0]["source_format"] == "start_anchored_think"
        assert coordinator.submit(db, request)
        if case != "pii":
            altered = {
                **response,
                "thinking": [{**thinking, "text": "Bearer sk-proj-" + "y" * 48}],
            }
            with pytest.raises(ValueError, match="settlement_response_conflict"):
                coordinator.settle(db, replace(request, response_envelope=altered))
        # Retiring the locator must retain the same historical response meaning.
        row = db.get_message_by_id(assistant_id)
        assert db.update_message(assistant_id, {"content": "edited"}, row["version"])
        assert (
            ConsoleTraceNativeReader(db)._reconstruct_response(cursor, call) == captured
        )
        # A retired masked body cannot re-prove raw equality for a new handoff.
        with pytest.raises(ValueError, match="settlement_response_conflict"):
            coordinator.settle(db, request)
    elif case == "legacy":
        assert captured == {"role": "assistant", "content": "answer"}


@pytest.mark.asyncio
@pytest.mark.parametrize("replay_harness", [(True, "thinking")], indirect=True)
@pytest.mark.parametrize("cold", [False, True])
@pytest.mark.parametrize("history", ["include", "exclude"])
async def test_thinking_response_pii_masks_preserve_successor_message_projection(
    replay_harness,
    cold,
    history,
):
    harness = replay_harness
    snapshot = harness.controller.capture_policy_snapshot("session-1")
    await harness.controller.replace_conversation_trace_privacy(
        "session-1",
        capture_enabled=True,
        pii_redaction_enabled=True,
        expected_policy_revision=snapshot.policy_revision,
    )
    _, persisted = harness.store.set_session_thinking_history_policy(
        "session-1", history
    )
    assert persisted
    canary = "response.person@example.com"
    harness.reply_text = f"<think>{canary}</think>answer"
    previous = []
    for text in ("alias request", "ordinary successor"):
        if cold:
            harness.factory = ConsoleTraceBoundaryFactory(harness.database)
        result = await harness.controller.submit_draft(text, session_id="session-1")
        assert result.accepted and result.provider_started, (result, harness.failures)
        saved = harness.store.get_message(result.user_message_id)
        reader = ConsoleTraceNativeReader(harness.database)
        trace = reader.read_calls(saved.persisted_message_id)
        assert len(trace) == 1
        assert "messages_payload" in trace[0].capture.request
        assert canary not in json.dumps(trace[0].capture.request)
        assert canary not in json.dumps(trace[0].capture.response)
        assert "[PII omitted]" in json.dumps(trace[0].capture.response)
        previous.append((saved.persisted_message_id, trace))
        for message_id, original in previous:
            assert reader.read_calls(message_id) == original
    assert len(harness.entries) == 2
    if history == "exclude":
        assert canary not in repr(harness.entries[-1])
        assert trace[0].capture.request["messages_payload"][-2]["content"] == "answer"
    else:
        assert canary in repr(harness.entries[-1])
