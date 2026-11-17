"""Behavior at the fixed, presentation-only Console value boundary."""

import json
from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType

import pytest

from tldw_chatbook.Agents.approval_observation import ApprovalObservationIdentity
from tldw_chatbook.Chat.console_approval_feedback import ApprovalFeedback
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleActivityPresentation,
    ConsoleChatMessage,
    ConsoleCitationPhase,
    ConsoleCitationPresentation,
    ConsoleMessageRole,
    ConsoleRunState,
    ConsoleRunStatus,
    ConsoleVariant,
    ConsoleVariantSet,
    MessageAttachment,
    RawCliPresentation,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession
from tldw_chatbook.Chat.console_execution_projection_values import (
    decode_continuation_projection,
    decode_message_projection,
    decode_run_projection,
    decode_session_projection,
    encode_continuation_projection,
    encode_message_projection,
    encode_run_projection,
    encode_session_projection,
)
from tldw_chatbook.Chat.console_execution_protocol import ConsoleProtocolError
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.message_metadata import AgentChatStartMetadata, MessageMetadata
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.Provider_Contract.thinking_blocks import (
    DisplayableThinkingBlock,
    ProprietaryThinkingBlock,
    ThinkingEnvelope,
)
from tldw_chatbook.UI.Console_Modules.provider_continuation_recovery import (
    ProviderContinuationRecoveryState,
)


def _message():
    """Use real values: loss of ordered media/thinking/activity breaks rendering."""
    return ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="Answer é",
        id="assistant-1",
        persisted_message_id="saved-1",
        turn_id="turn-1",
        trace_turn_id="trace-1",
        variants=ConsoleVariantSet.from_generations(
            turn_id="turn-1",
            generations=[
                ConsoleVariant(id="v1", content="First"),
                ConsoleVariant(id="v2", content="Answer é"),
            ],
            selected_index=1,
        ),
        image_data=b"\x00\xff",
        image_mime_type="image/png",
        attachments=(
            MessageAttachment(b"\x00\xff", "image/png", "one.png", 0),
            MessageAttachment(None, "image/jpeg", "missing.jpg", 1),
            MessageAttachment(b"", "image/png", "empty.png", 2),
        ),
        thinking=ThinkingEnvelope(
            blocks=(
                DisplayableThinkingBlock(
                    "visible",
                    0,
                    "provider",
                    "model",
                    "chat",
                    "text",
                    "complete",
                    "Visible reasoning é",
                ),
                ProprietaryThinkingBlock(
                    "hidden", 1, "provider", "model", "chat", "opaque", "complete"
                ),
            )
        ),
        activity_presentation=ConsoleActivityPresentation(
            "tool",
            "Read file",
            "success",
            call_id="call-1",
            arguments="path=a",
            result_preview="Found",
            elapsed_seconds=0.5,
            approval_feedback=ApprovalFeedback(
                ApprovalObservationIdentity("session", "run", "round", 1, "call-1"),
                sequence=2,
                decision_state="accepted",
                selected_scope="approve_once",
                execution_state="backend_started",
                confirmed_backend_start=True,
            ),
        ),
        activity_round_ordinal=0,
        tool_output_full="Full result",
        tool_diff=("file.py", "old\n", "new\n"),
        citation_presentation=ConsoleCitationPresentation(
            ConsoleCitationPhase.SELECTED
        ),
        evidence_source_count=0,
        assistant_generation_state="complete",
        metadata=MessageMetadata(
            origin="hook", mixin_attempt_id="mix-1", mixin_snapshot_schema_version=1
        ),
        mixin_usage_count=0,
        mixin_replay_status="recorded",
        usage=ProviderUsage(
            uncached_input=11, output=7, provider="provider", model="model"
        ),
        raw_cli_presentation=RawCliPresentation(
            "invocation",
            "model",
            "exited",
            "echo ok",
            "pwsh",
            "C:/project",
            10.0,
            0.5,
            0,
            False,
            True,
        ),
    )


def test_roundtrip_retains_ordered_visible_values_and_detaches_every_container():
    message = _message()
    encoded = encode_message_projection(message)
    assert encoded["image_data"] == "AP8="
    assert [row["data"] for row in encoded["attachments"]] == ["AP8=", None, ""]
    assert "text" not in encoded["thinking"]["blocks"][1]
    decoded = decode_message_projection(json.loads(json.dumps(encoded)))
    assert decoded == message
    assert decoded is not message
    assert decoded.variants is not message.variants
    encoded["variants"]["variants"][1]["content"] = "tampered"
    encoded["attachments"][0]["display_name"] = "tampered"
    encoded["thinking"]["blocks"][0]["text"] = "tampered"
    decoded.variants.selected_index = 0
    assert message.variants.selected_index == 1
    assert decoded.variants.variants[1].content == "Answer é"
    assert decoded.attachments[0].display_name == "one.png"
    assert decoded.thinking.blocks[0].text == "Visible reasoning é"
    assert encode_message_projection(message)["variants"]["selected_index"] == 1


def test_private_message_and_metadata_values_are_never_serialized():
    message = _message()
    message.provider_continuation = object()
    message.opaque_thinking_json = "private raw reasoning"
    message.exchanges = (object(),)
    message.live_activity = "render-only"
    message.metadata = replace(
        message.metadata, engine="private-engine", provider="secret"
    )
    message.variants.variants[0] = replace(
        message.variants.variants[0],
        provider_continuation=object(),
        opaque_thinking_json="private variant",
    )
    encoded = encode_message_projection(message)
    assert set(encoded["metadata"]) == {
        "origin",
        "template_kind",
        "template_source",
        "mixin_snapshot_schema_version",
        "mixin_attempt_id",
        "agent_chat_start",
    }
    rendered = json.dumps(encoded)
    assert "private" not in rendered
    assert "secret" not in rendered
    assert "provider_continuation" not in rendered
    assert "exchanges" not in encoded
    assert "live_activity" not in encoded
    assert decode_message_projection(encoded).provider_continuation is None


def test_saved_machine_origin_retains_only_original_bounded_provenance():
    message = ConsoleChatMessage(
        ConsoleMessageRole.USER,
        "Earlier machine-authored request",
        metadata=MessageMetadata(
            origin="agent_chat_start",
            agent_chat_start=AgentChatStartMetadata(
                "attempt-1", "run-1", "conversation-1"
            ),
        ),
    )
    encoded = encode_message_projection(message)
    assert encoded["metadata"]["agent_chat_start"] == {
        "attempt_id": "attempt-1",
        "source_run_id": "run-1",
        "source_conversation_id": "conversation-1",
    }
    assert decode_message_projection(encoded).metadata == message.metadata
    encoded["metadata"]["agent_chat_start"] = None
    with pytest.raises(ConsoleProtocolError):
        decode_message_projection(encoded)


def test_original_live_variant_generation_state_survives_projection():
    message = ConsoleChatMessage(
        ConsoleMessageRole.ASSISTANT,
        "partial",
        status="streaming",
        assistant_generation_state="streaming",
    )
    decoded = decode_message_projection(encode_message_projection(message))
    assert decoded.status == "streaming"
    assert decoded.assistant_generation_state == "streaming"
    assert decoded.content == "partial"


@pytest.mark.parametrize("location", ["message", "metadata", "thinking"])
def test_decode_rejects_injected_private_fields_at_each_record_boundary(location):
    payload = encode_message_projection(_message())
    target = payload if location == "message" else payload[location]
    target["provider_continuation"] = {"secret": "must not be adopted"}
    with pytest.raises(ConsoleProtocolError):
        decode_message_projection(payload)


@pytest.mark.parametrize("bad", [object(), MappingProxyType({}), {"content": object()}])
def test_decode_requires_exact_data_records(bad):
    with pytest.raises(ConsoleProtocolError):
        decode_message_projection(bad)


@pytest.mark.parametrize("change", ["base64", "ordinal", "selected", "finite", "text"])
def test_decode_refuses_values_that_cannot_render_with_original_semantics(change):
    payload = encode_message_projection(_message())
    if change == "base64":
        payload["image_data"] = "AP9="  # Same bytes as AP8=, noncanonical pad bits.
    elif change == "ordinal":
        payload["thinking"]["blocks"][1]["round_ordinal"] = 0
    elif change == "selected":
        payload["variants"]["selected_index"] = 2
    elif change == "finite":
        payload["usage"]["transcription_seconds"] = float("nan")
    else:
        payload["thinking"]["blocks"][1]["text"] = "must not expose proprietary text"
    with pytest.raises(ConsoleProtocolError):
        decode_message_projection(payload)


def test_encode_refuses_unsupported_message_subclasses_and_generation_branches():
    class CustomMessage(ConsoleChatMessage):
        pass

    with pytest.raises(ConsoleProtocolError):
        encode_message_projection(CustomMessage(ConsoleMessageRole.USER, "custom"))
    for field in ("generation_metadata", "video_metadata"):
        message = _message()
        setattr(
            message, field, (object(),) if field == "generation_metadata" else object()
        )
        with pytest.raises(ConsoleProtocolError, match="unsupported"):
            encode_message_projection(message)


def test_thinking_encoded_limit_is_preserved_even_when_each_block_is_valid():
    message = _message()
    # Five individually legal 256 KiB blocks exceed the original 1 MiB envelope.
    message.thinking = ThinkingEnvelope(
        blocks=tuple(
            DisplayableThinkingBlock(
                str(index),
                index,
                "provider",
                "model",
                "chat",
                "text",
                "complete",
                "x" * (256 * 1024),
            )
            for index in range(5)
        )
    )
    with pytest.raises(ConsoleProtocolError):
        encode_message_projection(message)


def test_session_projection_is_frozen_detached_and_does_not_expose_owners():
    session = ConsoleChatSession(
        id="session-1",
        title="Chat",
        settings=ConsoleSessionSettings(
            provider="openai",
            model="example",
            temperature=0.25,
            source="user",
            pinned_prefill="",
            system_prompt=None,
        ),
        draft="unsent é",
        draft_revision=3,
        capture_enabled_override=None,
        next_capture_enabled=False,
    )
    encoded = encode_session_projection(session)
    decoded = decode_session_projection(encoded)
    assert decoded.id == "session-1"
    assert decoded.draft == "unsent é"
    assert decoded.draft_revision == 3
    assert decoded.settings == session.settings
    assert decoded.settings is not session.settings
    assert decoded.capture_enabled_override is None
    assert decoded.next_capture_enabled is False
    assert "_progress_owner_id" not in encoded
    assert "project_instruction_state" not in encoded
    encoded["settings"]["model"] = "changed"
    assert decoded.settings.model == "example"
    with pytest.raises(FrozenInstanceError):
        decoded.title = "changed"
    encoded["settings"]["source_authority"] = object()
    with pytest.raises(ConsoleProtocolError):
        decode_session_projection(encoded)


def test_run_and_safe_recovery_roundtrip_without_checkpoint_authority():
    state = ConsoleRunState(ConsoleRunStatus.BLOCKED, "Review the pending tool")
    assert encode_run_projection(state) == {
        "status": "blocked",
        "visible_copy": "Review the pending tool",
    }
    assert decode_run_projection(encode_run_projection(state)) == state
    recovery = ProviderContinuationRecoveryState(
        "assistant", 4, "ambiguous", "Review", False, False
    )
    encoded = encode_continuation_projection(recovery)
    assert encoded == {
        "message_id": "assistant",
        "message_version": 4,
        "mode": "ambiguous",
        "impact": "Review",
        "replay_available": False,
        "actions_enabled": False,
    }
    assert decode_continuation_projection(encoded) == recovery
    encoded["checkpoint"] = object()
    with pytest.raises(ConsoleProtocolError):
        decode_continuation_projection(encoded)
    with pytest.raises(ConsoleProtocolError):
        decode_run_projection({"status": "invented", "visible_copy": ""})
