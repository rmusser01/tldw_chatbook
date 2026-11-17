"""Closed, detached Console presentation values; never execution authority."""

from __future__ import annotations

import base64
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, NoReturn

from .console_execution_protocol import ConsoleProtocolError

if TYPE_CHECKING:
    from .console_chat_models import ConsoleChatMessage, ConsoleRunState
    from .console_chat_store import ConsoleChatSession
    from .console_session_settings import ConsoleSessionSettings
    from ..UI.Console_Modules.provider_continuation_recovery import (
        ProviderContinuationRecoveryState,
    )

type _Codec = Callable[[object, bool], object]
type _Schema = dict[str, _Codec]


def _fail(reason: str) -> NoReturn:
    # Never include a supplied key, body, path, or constructor exception text.
    raise ConsoleProtocolError(f"Console projection: {reason}") from None


def _text(value: object, _decode: bool) -> str:
    if type(value) is not str:
        _fail("expected text")
    try:
        value.encode("utf-8")
    except UnicodeError:
        _fail("invalid Unicode")
    return value


def _nonempty_text(value: object, decode: bool) -> str:
    result = _text(value, decode)
    if not result:
        _fail("expected nonempty text")
    return result


def _integer(value: object, _decode: bool) -> int:
    if type(value) is not int:
        _fail("expected integer")
    return value


def _count(value: object, decode: bool) -> int:
    result = _integer(value, decode)
    if result < 0:
        _fail("negative count or revision")
    return result


def _number(value: object, _decode: bool) -> int | float:
    if type(value) not in (int, float):
        _fail("expected number")
    if type(value) is float and not math.isfinite(value):
        _fail("nonfinite number")
    return value


def _duration(value: object, decode: bool) -> int | float:
    result = _number(value, decode)
    if result < 0:
        _fail("negative duration")
    return result


def _boolean(value: object, _decode: bool) -> bool:
    if type(value) is not bool:
        _fail("expected boolean")
    return value


def _optional(codec: _Codec) -> _Codec:
    return lambda value, decode: None if value is None else codec(value, decode)


def _enum(*choices: str) -> _Codec:
    def check(value: object, decode: bool) -> str:
        result = _text(value, decode)
        if result not in choices:
            _fail("unknown enum value")
        return result

    return check


def _enum_record(value: object, enum_type: type, decode: bool):
    if decode:
        text = _text(value, True)
        try:
            return enum_type(text)
        except ValueError:
            _fail("unknown enum value")
    if type(value) is not enum_type:
        _fail("unsupported enum class")
    return value.value


def _bytes(value: object, decode: bool) -> bytes | str:
    if not decode:
        if type(value) is not bytes:
            _fail("expected original bytes")
        return base64.b64encode(value).decode("ascii")
    text = _text(value, True)
    try:
        result = base64.b64decode(text, validate=True)
    except ValueError:
        _fail("malformed base64")
    if base64.b64encode(result).decode("ascii") != text:
        _fail("noncanonical base64")
    return result


def _shape(value: object, schema: _Schema) -> dict:
    if (
        type(value) is not dict
        or any(type(key) is not str for key in value)
        or value.keys() != schema.keys()
    ):
        _fail("unexpected or missing record fields")
    return value


def _construct(cls: type, values: dict):
    try:
        return cls(**values)
    except (ValueError, TypeError, OverflowError):
        _fail("invalid original value record")


def _record(value: object, cls: type, schema: _Schema, decode: bool):
    """Walk only an explicitly declared finite record, never dataclass fields."""
    if decode:
        raw = _shape(value, schema)
        return _construct(
            cls, {name: codec(raw[name], True) for name, codec in schema.items()}
        )
    if type(value) is not cls:
        _fail("unsupported value class")
    encoded = {
        name: codec(getattr(value, name), False) for name, codec in schema.items()
    }
    # Re-run original constructors: even a mutated frozen instance is untrusted.
    _record(encoded, cls, schema, True)
    return encoded


def _tuple(codec: _Codec) -> _Codec:
    def transform(value: object, decode: bool):
        expected = list if decode else tuple
        if type(value) is not expected:
            _fail("expected ordered array")
        result = [codec(item, decode) for item in value]
        return tuple(result) if decode else result

    return transform


def _role(value: object, decode: bool):
    from .console_chat_models import ConsoleMessageRole

    return _enum_record(value, ConsoleMessageRole, decode)


def _run_status(value: object, decode: bool):
    from .console_chat_models import ConsoleRunStatus

    return _enum_record(value, ConsoleRunStatus, decode)


def _citation_phase(value: object, decode: bool):
    from .console_chat_models import ConsoleCitationPhase

    return _enum_record(value, ConsoleCitationPhase, decode)


def _citation_notice(value: object, decode: bool):
    from .console_chat_models import ConsoleCitationNoticeCode

    return _enum_record(value, ConsoleCitationNoticeCode, decode)


def _attachment(value: object, decode: bool):
    from .console_chat_models import MessageAttachment

    return _record(value, MessageAttachment, _ATTACHMENT, decode)


def _usage(value: object, decode: bool):
    from .provider_usage import ProviderUsage

    return _record(value, ProviderUsage, _USAGE, decode)


def _citation(value: object, decode: bool):
    from .console_chat_models import ConsoleCitationPresentation

    return _record(value, ConsoleCitationPresentation, _CITATION, decode)


def _approval_identity(value: object, decode: bool):
    from ..Agents.approval_observation import ApprovalObservationIdentity

    return _record(value, ApprovalObservationIdentity, _APPROVAL_IDENTITY, decode)


def _approval_feedback(value: object, decode: bool):
    from .console_approval_feedback import ApprovalFeedback

    return _record(value, ApprovalFeedback, _APPROVAL_FEEDBACK, decode)


def _activity(value: object, decode: bool):
    from .console_chat_models import ConsoleActivityPresentation

    return _record(value, ConsoleActivityPresentation, _ACTIVITY, decode)


def _raw_cli(value: object, decode: bool):
    from .console_chat_models import RawCliPresentation

    return _record(value, RawCliPresentation, _RAW_CLI, decode)


def _metadata(value: object, decode: bool):
    from .message_metadata import MessageMetadata

    return _record(value, MessageMetadata, _METADATA, decode)


def _agent_chat_start(value: object, decode: bool):
    from .message_metadata import AgentChatStartMetadata

    return _record(value, AgentChatStartMetadata, _AGENT_CHAT_START, decode)


def _settings(value: object, decode: bool):
    from .console_session_settings import ConsoleSessionSettings

    return _record(value, ConsoleSessionSettings, _SETTINGS, decode)


def _thinking(value: object, decode: bool):
    from ..Provider_Contract.thinking_blocks import (
        DisplayableThinkingBlock,
        ProprietaryThinkingBlock,
        ThinkingEnvelope,
        dump_thinking_blocks_json,
        parse_thinking_blocks_json,
    )

    if decode:
        raw = _shape(value, {"version": _integer, "blocks": _text})
        _integer(raw["version"], True)
        if type(raw["blocks"]) is not list:
            _fail("expected thinking block array")
        blocks = []
        for block in raw["blocks"]:
            if type(block) is not dict:
                _fail("expected thinking block record")
            visibility = block.get("visibility")
            if type(visibility) is not str or visibility not in {
                "displayable",
                "proprietary",
            }:
                _fail("unsupported thinking visibility")
            schema = dict(_THINKING_BLOCK)
            if visibility == "displayable":
                schema["text"] = _text
            checked = _shape(block, schema)
            blocks.append(
                {name: codec(checked[name], True) for name, codec in schema.items()}
            )
        # Original parser owns version, ordering, individual and encoded bounds.
        try:
            return parse_thinking_blocks_json(
                json.dumps(
                    {"version": raw["version"], "blocks": blocks}, ensure_ascii=False
                )
            )
        except (TypeError, ValueError, OverflowError):
            _fail("invalid supported thinking envelope")
    if type(value) is not ThinkingEnvelope or type(value.blocks) is not tuple:
        _fail("unsupported thinking envelope class")
    if any(
        type(block) not in (DisplayableThinkingBlock, ProprietaryThinkingBlock)
        for block in value.blocks
    ):
        _fail("unsupported thinking block class")
    try:
        encoded = json.loads(dump_thinking_blocks_json(value))
    except (TypeError, ValueError, OverflowError):
        _fail("invalid supported thinking envelope")
    _thinking(encoded, True)
    return encoded


def _variants(value: object, decode: bool):
    from .console_chat_models import ConsoleVariant, ConsoleVariantSet

    if decode:
        raw = _shape(
            value, {"turn_id": _text, "selected_index": _count, "variants": _text}
        )
        turn_id = _text(raw["turn_id"], True)
        selected = _count(raw["selected_index"], True)
        if type(raw["variants"]) is not list:
            _fail("expected variant array")
        variants = [
            _record(item, ConsoleVariant, _VARIANT, True) for item in raw["variants"]
        ]
        try:
            return ConsoleVariantSet.from_generations(
                turn_id=turn_id, generations=variants, selected_index=selected
            )
        except (TypeError, ValueError):
            _fail("invalid variant selection")
    if type(value) is not ConsoleVariantSet or type(value.variants) is not list:
        _fail("unsupported variant set class")
    encoded = {
        "turn_id": _text(value.turn_id, False),
        "selected_index": _count(value.selected_index, False),
        "variants": [
            _record(item, ConsoleVariant, _VARIANT, False) for item in value.variants
        ],
    }
    _variants(encoded, True)
    return encoded


def _tool_diff(value: object, decode: bool):
    expected = list if decode else tuple
    if type(value) is not expected or len(value) != 3:
        _fail("expected three diff strings")
    result = [_text(item, decode) for item in value]
    return tuple(result) if decode else result


_TEXT_OR_NONE = _optional(_text)
_COUNT_OR_NONE = _optional(_count)
_SCOPE = _enum(
    "approve_once",
    "approve_session",
    "allow_matching",
    "always_allow",
    "raw_shell_session",
    "deny",
)
_ATTACHMENT = {
    "data": _optional(_bytes),
    "mime_type": _text,
    "display_name": _text,
    "position": _count,
}
_USAGE = {
    "uncached_input": _count,
    "cache_read": _count,
    "cache_write": _count,
    "output": _count,
    "audio_input": _count,
    "audio_output": _count,
    "transcription_seconds": _duration,
    "provider": _text,
    "model": _text,
    "partial": _boolean,
}
_CITATION = {
    "phase": _citation_phase,
    "notice_code": _optional(_citation_notice),
    "original_attempt_available": _boolean,
}
_APPROVAL_IDENTITY = {
    "session_id": _text,
    "run_id": _text,
    "round_id": _text,
    "revision": _count,
    "call_key": _text,
}
_APPROVAL_FEEDBACK = {
    "identity": _approval_identity,
    "sequence": _count,
    "decision_state": _enum(
        "pending", "received", "accepted", "timeout", "cancelled", "revoked"
    ),
    "selected_scope": _optional(_SCOPE),
    "grant_state": _enum("unknown", "applied", "not_applied", "failed"),
    "applied_scope": _optional(_SCOPE),
    "execution_state": _enum(
        "", "dispatch_started", "backend_started", "tool_completed", "model_wait"
    ),
    "terminal_outcome": _enum(
        "", "success", "failure", "blocked", "timeout", "cancelled", "unknown"
    ),
    "confirmed_backend_start": _boolean,
}
_ACTIVITY = {
    "kind": _enum(
        "thinking",
        "planning",
        "tool",
        "spawn",
        "tasks",
        "changes",
        "feedback",
        "warning",
        "activity",
    ),
    "label": _text,
    "status": _enum(
        "starting",
        "queued",
        "running",
        "awaiting_approval",
        "timed_out",
        "success",
        "blocked",
        "denied",
        "blocked_off",
        "blocked_kill_switch",
        "failed",
        "done",
        "live",
        "stopped",
        "unavailable",
    ),
    "call_id": _text,
    "arguments": _text,
    "result_preview": _TEXT_OR_NONE,
    "started_at_monotonic": _optional(_duration),
    "elapsed_seconds": _optional(_duration),
    "approval_feedback": _optional(_approval_feedback),
}
_RAW_CLI = {
    "invocation_id": _text,
    "caller": _enum("user", "model"),
    "lifecycle_state": _enum(
        "starting",
        "running",
        "stopping",
        "exited",
        "timed_out",
        "cancelled",
        "cleanup_unproven",
        "failed",
    ),
    "command": _text,
    "shell": _text,
    "cwd": _text,
    "started_at_monotonic": _optional(_duration),
    "elapsed_seconds": _duration,
    "exit_code": _optional(_integer),
    "truncated": _boolean,
    "cleanup_proven": _optional(_boolean),
}
_METADATA = {
    "origin": _text,
    "template_kind": _text,
    "template_source": _text,
    "mixin_snapshot_schema_version": _COUNT_OR_NONE,
    "mixin_attempt_id": _TEXT_OR_NONE,
    "agent_chat_start": _optional(_agent_chat_start),
}
_AGENT_CHAT_START = {
    "attempt_id": _text,
    "source_run_id": _text,
    "source_conversation_id": _text,
}
_THINKING_BLOCK = {
    "block_id": _text,
    "round_ordinal": _count,
    "provider": _text,
    "model": _text,
    "protocol": _text,
    "source_format": _text,
    "status": _enum("complete", "stopped", "failed"),
    "visibility": _enum("displayable", "proprietary"),
}
_VARIANT = {"id": _text, "content": _text}
# Match the existing settings codec, including provider-rebased blank samplers.
_SETTINGS = {
    "provider": _nonempty_text,
    "model": _TEXT_OR_NONE,
    "base_url": _TEXT_OR_NONE,
    "temperature": _optional(_number),
    "top_p": _optional(_number),
    "min_p": _optional(_number),
    "top_k": _optional(_integer),
    "max_tokens": _optional(_integer),
    "seed": _optional(_integer),
    "presence_penalty": _optional(_number),
    "frequency_penalty": _optional(_number),
    "reasoning_effort": _TEXT_OR_NONE,
    "reasoning_summary": _TEXT_OR_NONE,
    "verbosity": _TEXT_OR_NONE,
    "thinking_effort": _TEXT_OR_NONE,
    "thinking_budget_tokens": _optional(_integer),
    "streaming": _boolean,
    "character_label": _text,
    "system_prompt": _TEXT_OR_NONE,
    "source": _enum("derived", "user"),
    "pinned_prefill": _TEXT_OR_NONE,
    "persona_memory_mode": _optional(_enum("read_only", "read_write")),
}
_MESSAGE = {
    "id": _text,
    "role": _role,
    "content": _text,
    "status": _enum(
        "complete", "pending", "streaming", "stopped", "failed", "discarded"
    ),
    "persisted_message_id": _TEXT_OR_NONE,
    "turn_id": _TEXT_OR_NONE,
    "trace_turn_id": _TEXT_OR_NONE,
    "sibling_index": _count,
    "sibling_count": _count,
    "variants": _optional(_variants),
    "image_data": _optional(_bytes),
    "image_mime_type": _TEXT_OR_NONE,
    "attachment_label": _TEXT_OR_NONE,
    "attachments": _tuple(_attachment),
    "activity_presentation": _optional(_activity),
    "activity_round_ordinal": _COUNT_OR_NONE,
    "tool_output_full": _TEXT_OR_NONE,
    "tool_diff": _optional(_tool_diff),
    "change_review_run_id": _TEXT_OR_NONE,
    "citation_presentation": _optional(_citation),
    "evidence_source_count": _COUNT_OR_NONE,
    "assistant_generation_state": _optional(
        _enum(
            "accepted",
            "dispatch_started",
            "continuation_active",
            "complete",
            # Original _begin_variant_stream publishes this live-only value.
            "streaming",
            "stopped",
            "failed",
            "discarded",
        )
    ),
    "generation_projection_quarantined": _boolean,
    "mixin_usage_count": _COUNT_OR_NONE,
    "mixin_replay_status": _optional(_enum("legacy", "recorded", "unavailable")),
    "metadata": _optional(_metadata),
    "usage": _optional(_usage),
    "thinking": _optional(_thinking),
    "raw_cli_presentation": _optional(_raw_cli),
}


@dataclass(frozen=True, slots=True)
class ConsoleSessionView:
    """Inert session-row display values; no backend ownership or mutations."""

    id: str
    title: str
    ephemeral: bool
    capture_enabled_override: bool | None
    next_capture_enabled: bool | None
    workspace_id: str
    persisted_conversation_id: str | None
    incarnation_id: str
    conversation_binding_revision: int
    settings_revision: int
    settings: ConsoleSessionSettings | None
    draft: str
    draft_revision: int


_SESSION = {
    "id": _text,
    "title": _text,
    "ephemeral": _boolean,
    "capture_enabled_override": _optional(_boolean),
    "next_capture_enabled": _optional(_boolean),
    "workspace_id": _text,
    "persisted_conversation_id": _TEXT_OR_NONE,
    "incarnation_id": _text,
    "conversation_binding_revision": _count,
    "settings_revision": _count,
    "settings": _optional(_settings),
    "draft": _text,
    "draft_revision": _count,
}
_RUN = {"status": _run_status, "visible_copy": _text}
_CONTINUATION = {
    "message_id": _text,
    "message_version": _count,
    "mode": _enum("reload", "notice", "ambiguous", "remote", "local"),
    "impact": _text,
    "replay_available": _boolean,
    "actions_enabled": _boolean,
}


def encode_message_projection(message: ConsoleChatMessage) -> dict:
    """Extract exact display fields, refusing unsupported generation branches."""
    from .console_chat_models import ConsoleChatMessage

    if type(message) is not ConsoleChatMessage:
        _fail("unsupported message class")
    if message.generation_metadata:
        _fail("unsupported image generation projection")
    if message.video_metadata is not None:
        _fail("unsupported video generation projection")
    encoded = _record(message, ConsoleChatMessage, _MESSAGE, False)
    if encoded["sibling_count"] <= encoded["sibling_index"]:
        _fail("invalid sibling selection")
    return encoded


def decode_message_projection(value: object) -> ConsoleChatMessage:
    """Build a disposable rendering copy with backend-only fields defaulted."""
    from .console_chat_models import ConsoleChatMessage

    message = _record(value, ConsoleChatMessage, _MESSAGE, True)
    if message.sibling_count <= message.sibling_index:
        _fail("invalid sibling selection")
    return message


def validate_message_projection(value: object) -> None:
    """Check the same finite fields without building a transcript message.

    Nested original value records still validate their own invariants. The
    original ConsoleChatMessage has no post-init validation; its wire checks
    are the explicit field schema and sibling selection below.
    """
    from .console_chat_models import ConsoleChatMessage

    if hasattr(ConsoleChatMessage, "__post_init__"):
        _fail("message constructor validation changed")
    raw = _shape(value, _MESSAGE)
    for name, codec in _MESSAGE.items():
        codec(raw[name], True)
    if raw["sibling_count"] <= raw["sibling_index"]:
        _fail("invalid sibling selection")


def encode_run_projection(state: ConsoleRunState) -> dict:
    """Encode the existing run status and its visible copy."""
    from .console_chat_models import ConsoleRunState

    return _record(state, ConsoleRunState, _RUN, False)


def decode_run_projection(value: object) -> ConsoleRunState:
    """Decode only the original finite run status vocabulary."""
    from .console_chat_models import ConsoleRunState

    return _record(value, ConsoleRunState, _RUN, True)


def encode_session_projection(session: ConsoleChatSession) -> dict:
    """Extract one original session's explicit row facts without its graph."""
    from .console_chat_store import ConsoleChatSession

    if type(session) is not ConsoleChatSession:
        _fail("unsupported session class")
    encoded = {
        name: codec(getattr(session, name), False) for name, codec in _SESSION.items()
    }
    decode_session_projection(encoded)
    return encoded


def decode_session_projection(value: object) -> ConsoleSessionView:
    """Return a frozen presentation row, never a ConsoleChatSession."""
    return _record(value, ConsoleSessionView, _SESSION, True)


def encode_continuation_projection(state: ProviderContinuationRecoveryState) -> dict:
    """Encode the backend's already-derived safe recovery presentation."""
    from ..UI.Console_Modules.provider_continuation_recovery import (
        ProviderContinuationRecoveryState,
    )

    return _record(state, ProviderContinuationRecoveryState, _CONTINUATION, False)


def decode_continuation_projection(value: object) -> ProviderContinuationRecoveryState:
    """Decode recovery display values without a private checkpoint."""
    from ..UI.Console_Modules.provider_continuation_recovery import (
        ProviderContinuationRecoveryState,
    )

    return _record(value, ProviderContinuationRecoveryState, _CONTINUATION, True)
