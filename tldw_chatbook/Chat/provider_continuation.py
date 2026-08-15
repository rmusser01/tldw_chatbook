"""Bounded provider-neutral storage for durable tool continuation.

The canonical result ceiling is the existing provider-bound
``RunBudget.max_tool_result_chars`` default: 16,000 characters. Keeping the
number local avoids importing the agent runtime into this pure storage model.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal, cast
from urllib.parse import urlsplit

from tldw_chatbook.Chat.provider_endpoint_contract import (
    canonical_connection_identity,
)


ContinuationProvider = Literal["moonshot", "zai", "deepseek"]
ContinuationProtocol = Literal["chat_completions", "responses"]
ContinuationState = Literal["active", "complete"]
ContinuationCallState = Literal["pending", "executing", "completed", "failed"]

_MAX_PAYLOAD_BYTES = 8 * 1024 * 1024
_MAX_ROUNDS = 128
_MAX_CALLS = 128
_MAX_IDENTITY_BYTES = 4 * 1024
_MAX_ARGUMENT_BYTES = 1024 * 1024
_MAX_REASONING_BYTES = 4 * 1024 * 1024
_MAX_RESULT_CHARS = 16_000
_MAX_JSON_DEPTH = 32
_MAX_JSON_NODES = 100_000
_INVALID_MESSAGE = "Invalid continuation data."
_DISCARDED_WARNING = "Exact tool continuation was discarded."
_KIMI_K3_MODEL = "kimi-k3"
_FUNCTION_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{2,63}$")

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "checkpoint_revision",
        "provider",
        "protocol",
        "model",
        "api_base_url",
        "state",
        "rounds",
    }
)
_ROUND_KEYS = frozenset({"assistant_content", "reasoning_blocks", "calls"})
_CALL_KEYS = frozenset({"call_id", "name", "arguments", "state"})
_CALL_RESULT_KEYS = _CALL_KEYS | {"result"}
_PAIRINGS = frozenset(
    {
        ("moonshot", "chat_completions"),
        ("zai", "chat_completions"),
        ("deepseek", "chat_completions"),
        ("deepseek", "responses"),
    }
)
_CALL_STATES = frozenset({"pending", "executing", "completed", "failed"})
_TERMINAL_CALL_STATES = frozenset({"completed", "failed"})
_CANONICAL_ENCODER = json.JSONEncoder(
    ensure_ascii=False,
    separators=(",", ":"),
    allow_nan=False,
)


class ContinuationValidationError(ValueError):
    """Report that private continuation failed canonical validation."""


class ContinuationConflictError(ValueError):
    """Report a stale optimistic continuation revision."""


@dataclass(frozen=True, repr=False)
class ContinuationResult:
    """The exact capped result string sent back to a provider."""

    value: str

    def __repr__(self) -> str:
        return "ContinuationResult(<private>)"


@dataclass(frozen=True, repr=False)
class ContinuationCall:
    """One canonical function call and its durable lifecycle state."""

    call_id: str
    name: str
    arguments: str
    state: ContinuationCallState
    result: ContinuationResult | None = None

    def __repr__(self) -> str:
        return f"ContinuationCall(state={self.state!r}, private_fields=<redacted>)"


@dataclass(frozen=True, repr=False)
class ContinuationRound:
    """One ordered assistant output in a provider continuation."""

    assistant_content: str
    reasoning_blocks: tuple[str, ...]
    calls: tuple[ContinuationCall, ...]

    def __repr__(self) -> str:
        return f"ContinuationRound(calls={len(self.calls)}, private_fields=<redacted>)"


@dataclass(frozen=True, repr=False)
class ProviderContinuationCheckpoint:
    """Canonical V1 durable provider continuation."""

    schema_version: Literal[1]
    checkpoint_revision: int
    provider: ContinuationProvider
    protocol: ContinuationProtocol
    model: str
    api_base_url: str
    state: ContinuationState
    rounds: tuple[ContinuationRound, ...]

    def __repr__(self) -> str:
        return (
            "ProviderContinuationCheckpoint("
            f"schema_version={self.schema_version}, "
            f"checkpoint_revision={self.checkpoint_revision}, "
            f"provider={self.provider!r}, protocol={self.protocol!r}, "
            f"state={self.state!r}, rounds={len(self.rounds)}, "
            "private_fields=<redacted>)"
        )


@dataclass(frozen=True, repr=False)
class SafeContinuationRead:
    """A tolerant private-data read result for visible-message imports."""

    checkpoint: ProviderContinuationCheckpoint | None
    warning: str | None = None

    def __repr__(self) -> str:
        return (
            "SafeContinuationRead("
            f"checkpoint_present={self.checkpoint is not None}, warning={self.warning!r})"
        )


@dataclass(frozen=True, repr=False)
class ContinuationOwnerGroup:
    """One visible assistant owner and its immutable canonical rounds."""

    owner_message_id: str
    checkpoint: ProviderContinuationCheckpoint
    rounds: tuple[ContinuationRound, ...]

    def __repr__(self) -> str:
        return (
            "ContinuationOwnerGroup("
            f"owner_message_id={self.owner_message_id!r}, rounds={len(self.rounds)}, "
            "private_fields=<redacted>)"
        )


@dataclass(frozen=True, repr=False)
class ContinuationRestoreTarget:
    """Current provider resolution that must exactly match a checkpoint."""

    provider: str
    model: str
    protocol: str
    api_base_url: str

    def __repr__(self) -> str:
        return (
            "ContinuationRestoreTarget("
            f"provider={self.provider!r}, protocol={self.protocol!r}, "
            "private_fields=<redacted>)"
        )


class _InvalidContinuation(Exception):
    pass


def _fail() -> None:
    raise _InvalidContinuation


def _strict_json_loads(value: str) -> object:
    def reject_constant(_value: str) -> None:
        _fail()

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                _fail()
            result[key] = item
        return result

    return json.loads(
        value,
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )


def _exact_mapping(value: object, keys: frozenset[str]) -> Mapping[str, object]:
    if type(value) is not dict or set(cast(dict[object, object], value)) != keys:
        _fail()
    return cast(Mapping[str, object], value)


def _exact_list(value: object) -> list[object]:
    if type(value) is not list:
        _fail()
    return cast(list[object], value)


def _exact_string(value: object, *, nonblank: bool = False) -> str:
    if type(value) is not str:
        _fail()
    text = cast(str, value)
    if nonblank and not text.strip():
        _fail()
    return text


def _bounded_utf8(value: str, maximum: int) -> int:
    if len(value) > maximum:
        _fail()
    byte_count = len(value.encode("utf-8"))
    if byte_count > maximum:
        _fail()
    return byte_count


def _validate_base_url(value: str) -> None:
    if value != value.strip() or any(character.isspace() for character in value):
        _fail()
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        _fail()
    parsed.port


def _json_shape(value: object) -> tuple[int, int]:
    """Return finite JSON node count and depth using a bounded iterative walk."""
    nodes = 0
    maximum_depth = 0
    pending: list[tuple[object, int]] = [(value, 1)]
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            _fail()
        maximum_depth = max(maximum_depth, depth)
        if type(item) is dict:
            mapping = cast(dict[object, object], item)
            child_count = len(mapping) * 2
            if nodes + len(pending) + child_count > _MAX_JSON_NODES:
                _fail()
            for key, nested in mapping.items():
                pending.append((key, depth + 1))
                pending.append((nested, depth + 1))
        elif type(item) is list:
            sequence = cast(list[object], item)
            if nodes + len(pending) + len(sequence) > _MAX_JSON_NODES:
                _fail()
            for nested in sequence:
                pending.append((nested, depth + 1))
        elif type(item) is float:
            if not math.isfinite(item):
                _fail()
        elif item is not None and type(item) not in {str, int, bool}:
            _fail()
    return nodes, maximum_depth


def _parse_arguments(value: object) -> tuple[str, int, int]:
    arguments = _exact_string(value)
    _bounded_utf8(arguments, _MAX_ARGUMENT_BYTES)
    parsed = _strict_json_loads(arguments)
    if type(parsed) is not dict:
        _fail()
    nodes, depth = _json_shape(parsed)
    return arguments, nodes, depth


def _parse_call(value: object) -> tuple[ContinuationCall, int, int]:
    if type(value) is not dict:
        _fail()
    item = cast(Mapping[str, object], value)
    keys = set(item)
    state = _exact_string(item.get("state"))
    expected_keys = _CALL_RESULT_KEYS if state in _TERMINAL_CALL_STATES else _CALL_KEYS
    if keys != expected_keys or state not in _CALL_STATES:
        _fail()

    call_id = _exact_string(item["call_id"], nonblank=True)
    name = _exact_string(item["name"])
    _bounded_utf8(call_id, _MAX_IDENTITY_BYTES)
    if _FUNCTION_NAME.fullmatch(name) is None:
        _fail()
    arguments, argument_nodes, argument_depth = _parse_arguments(item["arguments"])

    result: ContinuationResult | None = None
    if state in _TERMINAL_CALL_STATES:
        result_text = _exact_string(item["result"])
        if len(result_text) > _MAX_RESULT_CHARS:
            _fail()
        result = ContinuationResult(result_text)

    return (
        ContinuationCall(
            call_id=call_id,
            name=name,
            arguments=arguments,
            state=cast(ContinuationCallState, state),
            result=result,
        ),
        argument_nodes,
        argument_depth,
    )


def _parse_round(
    value: object,
    *,
    call_ids: set[str],
    remaining_calls: int,
) -> tuple[ContinuationRound, int, int, int, int]:
    item = _exact_mapping(value, _ROUND_KEYS)
    assistant_content = _exact_string(item["assistant_content"])
    _bounded_utf8(assistant_content, _MAX_PAYLOAD_BYTES)

    raw_reasoning = _exact_list(item["reasoning_blocks"])
    if len(raw_reasoning) > _MAX_JSON_NODES:
        _fail()
    reasoning: list[str] = []
    reasoning_bytes = 0
    for block_value in raw_reasoning:
        block = _exact_string(block_value)
        reasoning_bytes += _bounded_utf8(
            block,
            _MAX_REASONING_BYTES - reasoning_bytes,
        )
        reasoning.append(block)

    raw_calls = _exact_list(item["calls"])
    if len(raw_calls) > remaining_calls:
        _fail()
    calls: list[ContinuationCall] = []
    argument_nodes = 0
    argument_depth = 0
    for raw_call in raw_calls:
        call, call_argument_nodes, call_argument_depth = _parse_call(raw_call)
        if call.call_id in call_ids:
            _fail()
        call_ids.add(call.call_id)
        calls.append(call)
        argument_nodes += call_argument_nodes
        argument_depth = max(argument_depth, call_argument_depth)
        if argument_nodes > _MAX_JSON_NODES:
            _fail()

    return (
        ContinuationRound(
            assistant_content=assistant_content,
            reasoning_blocks=tuple(reasoning),
            calls=tuple(calls),
        ),
        len(calls),
        reasoning_bytes,
        argument_nodes,
        argument_depth,
    )


def _checkpoint_value(checkpoint: ProviderContinuationCheckpoint) -> dict[str, object]:
    rounds: list[dict[str, object]] = []
    for round_ in checkpoint.rounds:
        calls: list[dict[str, object]] = []
        for call in round_.calls:
            value: dict[str, object] = {
                "call_id": call.call_id,
                "name": call.name,
                "arguments": call.arguments,
                "state": call.state,
            }
            if call.result is not None:
                value["result"] = call.result.value
            calls.append(value)
        rounds.append(
            {
                "assistant_content": round_.assistant_content,
                "reasoning_blocks": list(round_.reasoning_blocks),
                "calls": calls,
            }
        )
    return {
        "schema_version": checkpoint.schema_version,
        "checkpoint_revision": checkpoint.checkpoint_revision,
        "provider": checkpoint.provider,
        "protocol": checkpoint.protocol,
        "model": checkpoint.model,
        "api_base_url": checkpoint.api_base_url,
        "state": checkpoint.state,
        "rounds": rounds,
    }


def _canonical_dump(checkpoint: ProviderContinuationCheckpoint) -> str:
    return _CANONICAL_ENCODER.encode(_checkpoint_value(checkpoint))


def _validate_canonical_size(checkpoint: ProviderContinuationCheckpoint) -> None:
    total_bytes = 0
    for chunk in _CANONICAL_ENCODER.iterencode(_checkpoint_value(checkpoint)):
        total_bytes += _bounded_utf8(chunk, _MAX_PAYLOAD_BYTES - total_bytes)


def _parse_checkpoint(
    checkpoint: ProviderContinuationCheckpoint,
) -> ProviderContinuationCheckpoint:
    try:
        return _parse_value(_checkpoint_value(checkpoint))
    except Exception:
        pass
    raise ContinuationValidationError(_INVALID_MESSAGE) from None


def _parse_value(value: object) -> ProviderContinuationCheckpoint:
    if type(value) is str:
        raw = cast(str, value)
        _bounded_utf8(raw, _MAX_PAYLOAD_BYTES)
        value = _strict_json_loads(raw)
    payload_nodes, _ = _json_shape(value)
    item = _exact_mapping(value, _TOP_LEVEL_KEYS)

    if type(item["schema_version"]) is not int or item["schema_version"] != 1:
        _fail()
    revision = item["checkpoint_revision"]
    if type(revision) is not int or cast(int, revision) <= 0:
        _fail()

    provider = _exact_string(item["provider"])
    protocol = _exact_string(item["protocol"])
    if (provider, protocol) not in _PAIRINGS:
        _fail()
    model = _exact_string(item["model"], nonblank=True)
    api_base_url = _exact_string(item["api_base_url"], nonblank=True)
    _bounded_utf8(model, _MAX_IDENTITY_BYTES)
    _bounded_utf8(api_base_url, _MAX_IDENTITY_BYTES)
    _validate_base_url(api_base_url)
    state = _exact_string(item["state"])
    if state not in {"active", "complete"}:
        _fail()

    raw_rounds = _exact_list(item["rounds"])
    if not raw_rounds or len(raw_rounds) > _MAX_ROUNDS:
        _fail()

    call_ids: set[str] = set()
    rounds: list[ContinuationRound] = []
    total_calls = 0
    total_reasoning_bytes = 0
    total_argument_nodes = 0
    maximum_argument_depth = 0
    for raw_round in raw_rounds:
        round_, calls, reasoning_bytes, argument_nodes, argument_depth = _parse_round(
            raw_round,
            call_ids=call_ids,
            remaining_calls=_MAX_CALLS - total_calls,
        )
        rounds.append(round_)
        total_calls += calls
        total_reasoning_bytes += reasoning_bytes
        total_argument_nodes += argument_nodes
        maximum_argument_depth = max(maximum_argument_depth, argument_depth)
        if (
            total_reasoning_bytes > _MAX_REASONING_BYTES
            or payload_nodes + total_argument_nodes > _MAX_JSON_NODES
            or maximum_argument_depth > _MAX_JSON_DEPTH
        ):
            _fail()

    for index, round_ in enumerate(rounds):
        if round_.calls:
            if index != len(rounds) - 1 and any(
                call.state not in _TERMINAL_CALL_STATES for call in round_.calls
            ):
                _fail()
            if state == "complete" and any(
                call.state not in _TERMINAL_CALL_STATES for call in round_.calls
            ):
                _fail()
            continue
        if (
            provider != "moonshot"
            or state != "complete"
            or index != len(rounds) - 1
            or model != _KIMI_K3_MODEL
            or not any(block.strip() for block in round_.reasoning_blocks)
        ):
            _fail()
    if (
        provider == "moonshot"
        and state == "complete"
        and model == _KIMI_K3_MODEL
        and rounds[-1].calls
    ):
        _fail()

    checkpoint = ProviderContinuationCheckpoint(
        schema_version=1,
        checkpoint_revision=cast(int, revision),
        provider=cast(ContinuationProvider, provider),
        protocol=cast(ContinuationProtocol, protocol),
        model=model,
        api_base_url=api_base_url,
        state=cast(ContinuationState, state),
        rounds=tuple(rounds),
    )
    _validate_canonical_size(checkpoint)
    return checkpoint


def parse_provider_continuation_json(
    value: object,
) -> ProviderContinuationCheckpoint:
    """Strictly validate one JSON string or decoded canonical V1 value.

    Args:
        value: JSON text or its decoded mapping.

    Returns:
        An immutable canonical checkpoint.

    Raises:
        ContinuationValidationError: If the value is not canonical V1 data.
    """
    try:
        return _parse_value(value)
    except Exception:  # the public boundary must not retain private parser context
        pass
    raise ContinuationValidationError(_INVALID_MESSAGE) from None


def read_provider_continuation_json(value: object) -> SafeContinuationRead:
    """Tolerantly read imported private data without affecting visible content."""
    if value is None:
        return SafeContinuationRead(checkpoint=None)
    try:
        checkpoint = _parse_value(value)
    except Exception:  # the public boundary must not retain private parser context
        return SafeContinuationRead(
            checkpoint=None,
            warning=_DISCARDED_WARNING,
        )
    return SafeContinuationRead(checkpoint=checkpoint)


def dump_provider_continuation_json(
    checkpoint: ProviderContinuationCheckpoint | None,
) -> str | None:
    """Serialize a checkpoint into deterministic canonical JSON."""
    if checkpoint is None:
        return None
    try:
        validated = _parse_value(_checkpoint_value(checkpoint))
        return _canonical_dump(validated)
    except Exception:  # the public boundary must not retain private parser context
        pass
    raise ContinuationValidationError(_INVALID_MESSAGE) from None


def transition_provider_call(
    checkpoint: ProviderContinuationCheckpoint,
    *,
    call_id: str,
    expected_revision: int,
    target: ContinuationCallState,
    result: ContinuationResult | None = None,
) -> ProviderContinuationCheckpoint:
    """Apply one legal optimistic call-state transition.

    ``pending -> failed`` is the atomic review-refusal path: it records an
    exact result for a call that performed no external side effect. All other
    terminal transitions still require ``executing`` first.

    Args:
        checkpoint: Current canonical checkpoint.
        call_id: Globally unique call to update.
        expected_revision: Revision read by the caller.
        target: Desired next state.
        result: Exact provider-bound result for a terminal transition.

    Returns:
        A new checkpoint, or ``checkpoint`` for an exact terminal replay.

    Raises:
        ContinuationConflictError: If ``expected_revision`` is stale.
        ContinuationValidationError: If the transition is invalid.
    """
    _parse_checkpoint(checkpoint)

    if type(expected_revision) is not int:
        raise ContinuationValidationError(_INVALID_MESSAGE) from None
    if expected_revision != checkpoint.checkpoint_revision:
        raise ContinuationConflictError("Continuation revision conflict.") from None
    if type(call_id) is not str or not call_id.strip():
        raise ContinuationValidationError(_INVALID_MESSAGE) from None
    if type(target) is not str or target not in _CALL_STATES:
        raise ContinuationValidationError(_INVALID_MESSAGE) from None

    terminal_target = target in _TERMINAL_CALL_STATES
    if terminal_target:
        if type(result) is not ContinuationResult or type(result.value) is not str:
            raise ContinuationValidationError(_INVALID_MESSAGE) from None
        if len(result.value) > _MAX_RESULT_CHARS:
            raise ContinuationValidationError(_INVALID_MESSAGE) from None
    elif result is not None:
        raise ContinuationValidationError(_INVALID_MESSAGE) from None

    location: tuple[int, int] | None = None
    current: ContinuationCall | None = None
    for round_index, round_ in enumerate(checkpoint.rounds):
        for call_index, call in enumerate(round_.calls):
            if call.call_id == call_id:
                location = (round_index, call_index)
                current = call
                break
        if current is not None:
            break
    if location is None or current is None:
        raise ContinuationValidationError(_INVALID_MESSAGE) from None

    if current.state in _TERMINAL_CALL_STATES:
        if current.state == target and current.result == result:
            return checkpoint
        raise ContinuationValidationError(_INVALID_MESSAGE) from None
    if not (
        (current.state == "pending" and target in {"executing", "failed"})
        or (current.state == "executing" and terminal_target)
    ):
        raise ContinuationValidationError(_INVALID_MESSAGE) from None

    round_index, call_index = location
    calls = list(checkpoint.rounds[round_index].calls)
    calls[call_index] = replace(current, state=target, result=result)
    rounds = list(checkpoint.rounds)
    rounds[round_index] = replace(rounds[round_index], calls=tuple(calls))
    updated = replace(
        checkpoint,
        checkpoint_revision=checkpoint.checkpoint_revision + 1,
        rounds=tuple(rounds),
    )
    try:
        return _parse_value(_checkpoint_value(updated))
    except Exception:
        pass
    raise ContinuationValidationError(_INVALID_MESSAGE) from None


def validate_continuation_restore(
    checkpoint: ProviderContinuationCheckpoint,
    target: ContinuationRestoreTarget,
) -> None:
    """Require current provider resolution to exactly match a checkpoint.

    Args:
        checkpoint: Private continuation proposed for restore.
        target: Current provider resolution.

    Raises:
        ContinuationValidationError: If either value is malformed.
        ContinuationConflictError: If any pinned field differs.
    """
    canonical = _parse_checkpoint(checkpoint)
    if type(target) is not ContinuationRestoreTarget or any(
        type(value) is not str
        for value in (
            target.provider,
            target.protocol,
            target.model,
            target.api_base_url,
        )
    ):
        raise ContinuationValidationError(_INVALID_MESSAGE) from None
    checkpoint_endpoint = canonical_connection_identity(
        canonical.provider, canonical.api_base_url
    )
    if checkpoint_endpoint is None:
        raise ContinuationValidationError(_INVALID_MESSAGE) from None
    if (
        target.provider,
        target.protocol,
        target.model,
    ) != (
        canonical.provider,
        canonical.protocol,
        canonical.model,
    ) or target.api_base_url not in {
        canonical.api_base_url,
        checkpoint_endpoint[1],
    }:
        raise ContinuationConflictError(
            "Continuation restore target mismatch."
        ) from None


def continuation_owner_group(
    visible_message: Mapping[str, Any],
    checkpoint: ProviderContinuationCheckpoint | None,
) -> ContinuationOwnerGroup:
    """Bind canonical private rounds to exactly one visible assistant owner.

    Args:
        visible_message: Existing visible message projection.
        checkpoint: Provider continuation owned by that message.

    Returns:
        A detached immutable canonical owner snapshot.

    Raises:
        ContinuationValidationError: If owner identity, role, or checkpoint is invalid.
    """
    if not isinstance(visible_message, Mapping) or checkpoint is None:
        raise ContinuationValidationError(_INVALID_MESSAGE) from None
    owner_message_id = visible_message.get("id")
    if (
        type(owner_message_id) is not str
        or not cast(str, owner_message_id).strip()
        or visible_message.get("role") != "assistant"
    ):
        raise ContinuationValidationError(_INVALID_MESSAGE) from None
    canonical = _parse_checkpoint(checkpoint)
    return ContinuationOwnerGroup(
        owner_message_id=cast(str, owner_message_id),
        checkpoint=canonical,
        rounds=canonical.rounds,
    )
