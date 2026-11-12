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
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution
from tldw_chatbook.Chat.console_semantic_revision import (
    project_semantic_revision_trace_message,
)
from tldw_chatbook.Chat.console_trace_models import FrozenTracePolicy, new_opaque_id
from tldw_chatbook.Chat.console_trace_native_reader import ConsoleTraceNativeReader
from tldw_chatbook.Chat.console_trace_redaction import BUILTIN_PII_RULESET_REVISION_ID
from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
from tldw_chatbook.Chat.console_trace_runtime import ConsoleTraceBoundaryFactory
from tldw_chatbook.Chat.console_trace_service import _response_projection_profile
from tldw_chatbook.Chat.console_trace_settlement import (
    ConsoleTraceSettlementCoordinator,
)
from tldw_chatbook.Chat.thinking_blocks import (
    DisplayableThinkingBlock,
    ProprietaryThinkingBlock,
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


@pytest.mark.parametrize("profile_version", [1, 2])
@pytest.mark.parametrize(
    "case",
    [
        "exact",
        "fragmented",
        "pii",
        "pii_structured",
        "pii_structured_mismatch",
        "pii_mismatch",
        "pii_unavailable",
        "pii_legacy_mismatch",
        "pii_legacy_unavailable",
        "pii_legacy_exact",
        "pii_legacy_exact_unavailable",
        "queued",
        "raw_mismatch",
        "raw_mismatch_structured",
        "provider",
        "model",
        "format",
        "status",
        "round",
        "legacy",
        "ignored",
        "malformed",
        "unknown_format",
        "protocol",
        "type",
        "proprietary",
        "profile_extra",
        "profile_missing",
        "profile_version_bool",
        "profile_version_float",
        "profile_version_unknown",
        "profile_round_trip_bool",
        "profile_round_trip_unknown",
        "profile_protocol",
        "profile_reordered",
        "profile_widened",
        "profile_scalar_formats",
        "profile_empty_formats",
        "profile_null_formats",
        "profile_unknown_provider",
        "profile_blank_model",
        "profile_nonstring_model",
        "kimi_tagged",
        "kimi_structured",
    ],
)
def test_saved_thinking_response_projection_requires_exact_evidence(
    db, case, monkeypatch, profile_version
):
    repository = ConsoleTraceRepository()
    coordinator = ConsoleTraceSettlementCoordinator(repository)
    profile = dict(PROFILE)
    if profile_version == 2:
        profile = {
            "version": 2,
            "thinking_stream_disposition": "displayable",
            "thinking_round_trip_version": 1,
            "provider": "llama_cpp",
            "model": "unsloth/Qwen3.8-27B-GGUF:Q8_0",
            "protocol": "chat_completions",
            "source_formats": ["start_anchored_think", "reasoning_content"],
        }
    if case.startswith("kimi"):
        profile["model"] = "Local-KiMi-Test"
        if profile_version == 2:
            profile["source_formats"] = ["start_anchored_think"]
    if case == "proprietary":
        profile.update(
            thinking_stream_disposition="proprietary",
            provider="moonshot",
            model="provider-model",
        )
        if profile_version == 1:
            profile["source_format"] = "reasoning_content"
    provider, model = profile["provider"], profile["model"]
    if "legacy" in case:
        profile = None
    mutations = {
        "profile_extra": {"unexpected": True},
        "profile_version_bool": {"version": True},
        "profile_version_float": {"version": float(profile_version)},
        "profile_version_unknown": {"version": 3},
        "profile_round_trip_bool": {"thinking_round_trip_version": True},
        "profile_round_trip_unknown": {"thinking_round_trip_version": 2},
        "profile_protocol": {"protocol": "unknown"},
        "profile_reordered": {
            "source_formats": ["reasoning_content", "start_anchored_think"]
        },
        "profile_widened": {
            "source_formats": ["start_anchored_think", "reasoning_content", "reasoning"]
        },
        "profile_scalar_formats": {"source_formats": "start_anchored_think"},
        "profile_empty_formats": {"source_formats": []},
        "profile_null_formats": {"source_formats": None},
        "profile_unknown_provider": {"provider": "ollama"},
        "profile_blank_model": {"model": "  "},
        "profile_nonstring_model": {"model": None},
    }
    if case in mutations:
        profile.update(mutations[case])
    elif case == "profile_missing":
        profile.pop("model")
    if profile_version == 2 and case == "profile_unknown_provider":
        provider = "ollama"
    if case == "ignored":
        profile["thinking_stream_disposition"] = "ignored"
    conversation_id, _segment, call_id = _call(
        db,
        repository,
        response_projection=profile,
        provider=provider,
        model=model,
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
        provider=provider,
        model=model,
        protocol="chat_completions",
        source_format="start_anchored_think",
        status="complete",
        text=secret,
    )
    if case in {"provider", "model", "protocol"}:
        block = replace(block, **{case: "different"})
    elif case == "format" or "structured" in case:
        block = replace(block, source_format="reasoning_content")
    elif case == "unknown_format":
        block = replace(block, source_format="reasoning")
    elif case == "status":
        block = replace(block, status="stopped")
    elif case == "round":
        block = replace(block, round_ordinal=1)
    if case in {"pii_mismatch", "pii_legacy_mismatch", "pii_structured_mismatch"}:
        block = replace(block, text="other.person@example.com")
    if case in {"type", "proprietary"}:
        block = ProprietaryThinkingBlock(
            block_id=block.block_id,
            round_ordinal=0,
            provider=provider,
            model=model,
            protocol="chat_completions",
            source_format="reasoning_content",
            status="complete",
        )
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
    thinking.update(provider=provider, model=model)
    if case == "format" or "structured" in case:
        thinking["source_format"] = "reasoning_content"
    thinking["text"] = (
        "Bearer sk-proj-" + "y" * 48 if case.startswith("raw_mismatch") else secret
    )
    response = {"role": "assistant", "content": "answer", "thinking": [thinking]}
    if case == "proprietary":
        response.pop("thinking")
        response["proprietary_thinking_evidence"] = [
            {
                "provider": provider,
                "model": model,
                "protocol": "chat_completions",
                "source_format": "reasoning_content",
            }
        ]
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
    expects_revision = (
        case
        in {
            "exact",
            "fragmented",
            "pii",
            "queued",
            "legacy",
            "pii_legacy_exact",
            "kimi_tagged",
        }
        or (profile_version == 2 and case in {"format", "pii_structured"})
        or (profile_version == 1 and case == "proprietary")
    )
    assert link.link_kind == ("revision" if expects_revision else "artifact")
    assert link.verification_outcome == (
        "verified_equal" if expects_revision else "sanitized_artifact"
    )
    if expects_revision:
        revision = repository.get_semantic_revision(cursor, link.semantic_revision_id)
        assert revision.source_message_id == assistant_id
        assert revision.source_conversation_id == conversation_id
    captured = ConsoleTraceNativeReader(db)._reconstruct_response(cursor, call)
    assert secret not in json.dumps(captured)
    if case in {"pii", "pii_structured"} and expects_revision:
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
    if case in {"exact", "fragmented", "pii", "queued", "kimi_tagged"} or (
        profile_version == 2 and case in {"format", "pii_structured"}
    ):
        assert captured["content"] == "answer" and len(captured["thinking"]) == 1
        assert captured["thinking"][0]["source_format"] == thinking["source_format"]
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


@pytest.mark.bootstrap_profile
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


@pytest.mark.parametrize(
    "provider", ["llama_cpp", "local_llamacpp", "vllm", "local_vllm"]
)
@pytest.mark.parametrize("model", ["Qwen3", "Local-KiMi-Test", "", "  ", None, 12])
def test_new_local_profile_freezes_only_supported_parser_formats(provider, model):
    profile = _response_projection_profile(
        ConsoleProviderResolution(
            provider=provider,
            base_url="http://127.0.0.1:9099",
            ready=True,
            execution_key=provider,
            model=model,
            thinking_stream_disposition="displayable",
            thinking_round_trip_version=1,
        )
    )
    if not isinstance(model, str) or not model.strip():
        assert profile is None
        return
    assert profile == {
        "version": 2,
        "thinking_stream_disposition": "displayable",
        "thinking_round_trip_version": 1,
        "provider": provider,
        "model": model,
        "protocol": "chat_completions",
        "source_formats": ["start_anchored_think"]
        if model == "Local-KiMi-Test"
        else ["start_anchored_think", "reasoning_content"],
    }


@pytest.mark.parametrize(
    "provider", ["moonshot", "zai", "fireworks", "openai", "ollama", "LLAMA_CPP"]
)
@pytest.mark.parametrize("disposition", ["displayable", "proprietary", "ignored"])
def test_hosted_profiles_remain_v1_and_unknown_local_profiles_are_not_admitted(
    provider, disposition
):
    resolution = ConsoleProviderResolution(
        provider=provider,
        execution_key=provider,
        base_url="http://127.0.0.1:9099",
        ready=True,
        model="provider-model",
        thinking_stream_disposition=disposition,
        thinking_round_trip_version=None if disposition == "ignored" else 1,
    )
    profile = _response_projection_profile(resolution)
    if provider not in {"moonshot", "zai", "fireworks"} or disposition != "proprietary":
        assert profile is None
        return
    assert profile == {
        "version": 1,
        "thinking_stream_disposition": "proprietary",
        "thinking_round_trip_version": 1,
        "provider": provider,
        "model": "provider-model",
        "protocol": "chat_completions",
        "source_format": "reasoning_content",
    }
