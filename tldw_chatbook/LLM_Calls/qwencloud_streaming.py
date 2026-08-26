"""Record-aware streaming normalization for the QwenCloud adapter."""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Never, cast

import requests

from tldw_chatbook.Chat.Chat_Deps import ChatProviderError
from tldw_chatbook.LLM_Calls.hosted_chat_streaming import SSERecordDecoder

if TYPE_CHECKING:
    from tldw_chatbook.LLM_Calls.qwencloud import QwenCloudAPIMode

_JSON_DECODE_FAILED = object()
_STREAM_END = object()
_STREAM_READ_FAILED = object()
_STREAM_TRANSLATION_FAILED = object()
_FINGERPRINT_FAILED = object()
_MAPPING_COPY_FAILED = object()

# These ceilings are deliberately far above supported model outputs while
# bounding memory and CPU controlled by a provider stream.
_MAX_SSE_LINE_CHARS = 16 * 1024 * 1024
_MAX_SSE_RECORD_CHARS = 16 * 1024 * 1024
# Bound Python object overhead independently of decoded character ceilings.
_MAX_SSE_LINE_SEGMENTS = 65_536
_MAX_SSE_DATA_LINES = 65_536
_MAX_OUTPUT_CHARS = 32 * 1024 * 1024
# Provider identities and function names are retained for stream correlation.
_MAX_METADATA_CHARS = 4 * 1024
_MAX_STREAM_EVENTS = 200_000
_MAX_TRACKED_SEQUENCES = 200_000
_MAX_JSON_DEPTH = 128
_MAX_JSON_NODES = 1_000_000


def _stream_error(message: str) -> ChatProviderError:
    return ChatProviderError(provider="qwencloud", message=message, status_code=502)


def _raise_malformed(message: str) -> Never:
    raise _stream_error(message)


def _best_effort_close(resource: Any) -> None:
    """Attempt one close without exposing provider-controlled cleanup failures."""
    try:
        resource.close()
    except Exception:
        pass


def _reject_json_constant(_value: str) -> Never:
    raise ValueError


def _strict_json_loads(value: str) -> Any:
    try:
        decoded = json.loads(value, parse_constant=_reject_json_constant)
    except (RecursionError, TypeError, ValueError):
        return _JSON_DECODE_FAILED
    if not _json_shape_is_safe(decoded):
        return _JSON_DECODE_FAILED
    return decoded


def _json_shape_is_safe(value: Any) -> bool:
    """Validate JSON types, depth, and nodes iteratively before recursive work."""
    stack: list[tuple[Any, int]] = [(value, 1)]
    scheduled_nodes = 1
    try:
        while stack:
            node, depth = stack.pop()
            if depth > _MAX_JSON_DEPTH:
                return False
            if type(node) is dict:
                for key, child in cast(dict[Any, Any], node).items():
                    if not isinstance(key, str):
                        return False
                    scheduled_nodes += 1
                    if scheduled_nodes > _MAX_JSON_NODES:
                        return False
                    stack.append((child, depth + 1))
                continue
            if type(node) is list:
                for child in cast(list[Any], node):
                    scheduled_nodes += 1
                    if scheduled_nodes > _MAX_JSON_NODES:
                        return False
                    stack.append((child, depth + 1))
                continue
            if node is None or isinstance(node, (str, bool)):
                continue
            if isinstance(node, int) and not isinstance(node, bool):
                continue
            if isinstance(node, float) and math.isfinite(node):
                continue
            return False
    except (RecursionError, TypeError, ValueError):
        return False
    return True


def _canonical_json_digest(value: Any) -> bytes | object:
    try:
        canonical = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError):
        return _FINGERPRINT_FAILED
    return hashlib.sha256(canonical).digest()


def _copy_top_mapping(value: Mapping[str, Any]) -> dict[str, Any] | object:
    try:
        return dict(value)
    except Exception:
        return _MAPPING_COPY_FAILED


def _required_index(event: Mapping[str, Any], name: str) -> int:
    value = event.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        _raise_malformed("QwenCloud stream index is malformed.")
    return value


def _required_string(
    source: Mapping[str, Any], name: str, *, allow_empty: bool = False
) -> str:
    value = source.get(name)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        _raise_malformed("QwenCloud stream event identity is malformed.")
    return value


def _metadata_characters(*values: str) -> int:
    if any(len(value) > _MAX_METADATA_CHARS for value in values):
        _raise_malformed("QwenCloud stream metadata limit was exceeded.")
    return sum(len(value) for value in values)


@dataclass
class _OutputItemState:
    item_id: str
    item_type: str
    done_status: str | None = None


@dataclass
class _TextPartState:
    item_id: str
    fragments: list[str] = field(default_factory=list)
    delta_count: int = 0
    final_text: str | None = None


@dataclass
class _FunctionCallState:
    item_id: str
    call_id: str
    name: str
    tool_index: int
    argument_fragments: list[str] = field(default_factory=list)
    final_arguments: str | None = None


@dataclass(frozen=True)
class _ChatToolState:
    call_id: str
    name: str


@dataclass
class _ChatChoiceState:
    tools: dict[int, _ChatToolState]
    call_ids: set[str]
    terminal_reason: str | None = None


class QwenResponsesStreamTranslator:
    """Translate stateful QwenCloud Responses events into chat deltas."""

    def __init__(self) -> None:
        self._seen_sequences: dict[int, bytes] = {}
        self._highest_sequence = -1
        self._output_items: dict[int, _OutputItemState] = {}
        self._text_parts: dict[tuple[int, int], _TextPartState] = {}
        self._function_calls: dict[int, _FunctionCallState] = {}
        self._call_ids: set[str] = set()
        self._output_chars = 0
        self._terminal = False

    def feed(self, event: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
        """Consume one decoded Responses event and return normalized chunks."""
        if not isinstance(event, Mapping):
            _raise_malformed("QwenCloud stream event must be an object.")
        event_copy = _copy_top_mapping(event)
        if event_copy is _MAPPING_COPY_FAILED or not _json_shape_is_safe(event_copy):
            _raise_malformed("QwenCloud stream event is malformed.")
        event = cast(dict[str, Any], event_copy)
        sequence = event.get("sequence_number")
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
            _raise_malformed("QwenCloud stream sequence number is malformed.")
        digest = _canonical_json_digest(event)
        if digest is _FINGERPRINT_FAILED:
            _raise_malformed("QwenCloud stream event is malformed.")
        previous = self._seen_sequences.get(sequence)
        if previous is not None:
            if previous == digest:
                return ()
            _raise_malformed("QwenCloud stream sequence replay conflicts.")
        if self._terminal:
            _raise_malformed("QwenCloud stream event arrived after the terminal event.")
        if sequence <= self._highest_sequence:
            _raise_malformed("QwenCloud stream sequence number decreased.")
        if len(self._seen_sequences) >= _MAX_TRACKED_SEQUENCES:
            _raise_malformed("QwenCloud stream sequence limit was exceeded.")
        self._seen_sequences[sequence] = cast(bytes, digest)
        self._highest_sequence = sequence

        event_type = event.get("type")
        if not isinstance(event_type, str) or not event_type:
            _raise_malformed("QwenCloud stream event type is malformed.")
        if event_type in {"response.created", "response.in_progress"}:
            response = event.get("response")
            if not isinstance(response, Mapping):
                _raise_malformed("QwenCloud stream lifecycle event is malformed.")
            return ()
        if event_type == "response.output_item.added":
            return self._handle_output_item_added(event)
        if event_type == "response.content_part.added":
            return self._handle_content_part_added(event)
        if event_type == "response.output_text.delta":
            return self._handle_text_delta(event)
        if event_type == "response.output_text.done":
            return self._handle_text_done(event)
        if event_type == "response.function_call_arguments.delta":
            return self._handle_arguments_delta(event)
        if event_type == "response.function_call_arguments.done":
            return self._handle_arguments_done(event)
        if event_type == "response.content_part.done":
            return self._handle_content_part_done(event)
        if event_type == "response.output_item.done":
            return self._handle_output_item_done(event)
        if event_type in {
            "response.completed",
            "response.incomplete",
            "response.failed",
            "response.cancelled",
        }:
            return self._handle_terminal(event_type, event)
        _raise_malformed("QwenCloud stream event type is unsupported.")

    def finish(self) -> tuple[dict[str, Any], ...]:
        """Validate that the Responses stream supplied one terminal event."""
        if not self._terminal:
            _raise_malformed("QwenCloud stream terminal event is missing.")
        return ()

    def _handle_output_item_added(
        self, event: Mapping[str, Any]
    ) -> tuple[dict[str, Any], ...]:
        output_index = _required_index(event, "output_index")
        item = event.get("item")
        if not isinstance(item, Mapping):
            _raise_malformed("QwenCloud stream output item is malformed.")
        item_id = _required_string(item, "id")
        item_type = _required_string(item, "type")
        if item.get("status") != "in_progress":
            _raise_malformed("QwenCloud stream output item status is malformed.")
        if output_index in self._output_items:
            _raise_malformed("QwenCloud stream output index was reused.")
        if any(state.item_id == item_id for state in self._output_items.values()):
            _raise_malformed("QwenCloud stream output item identity was reused.")
        if item_type not in {"message", "reasoning", "function_call"}:
            _raise_malformed("QwenCloud stream output item type is unsupported.")
        if item_type == "function_call":
            call_id = _required_string(item, "call_id")
            name = _required_string(item, "name")
            arguments = _required_string(item, "arguments", allow_empty=True)
            if call_id in self._call_ids:
                _raise_malformed("QwenCloud stream function-call ID was reused.")
            self._reserve_output(
                _metadata_characters(item_id, call_id, name) + len(arguments)
            )
            self._output_items[output_index] = _OutputItemState(item_id, item_type)
            self._call_ids.add(call_id)
            call_state = _FunctionCallState(
                item_id=item_id,
                call_id=call_id,
                name=name,
                tool_index=output_index,
                argument_fragments=[arguments] if arguments else [],
            )
            self._function_calls[output_index] = call_state
            return (
                self._tool_chunk(
                    call_state,
                    arguments=arguments,
                    include_identity=True,
                ),
            )
        self._reserve_output(_metadata_characters(item_id))
        self._output_items[output_index] = _OutputItemState(item_id, item_type)
        return ()

    def _message_state(
        self, event: Mapping[str, Any]
    ) -> tuple[int, int, _OutputItemState]:
        output_index = _required_index(event, "output_index")
        content_index = _required_index(event, "content_index")
        item_id = _required_string(event, "item_id")
        item_state = self._output_items.get(output_index)
        if (
            item_state is None
            or item_state.item_type != "message"
            or item_state.item_id != item_id
        ):
            _raise_malformed("QwenCloud stream text event identity is inconsistent.")
        return output_index, content_index, item_state

    def _handle_content_part_added(
        self, event: Mapping[str, Any]
    ) -> tuple[dict[str, Any], ...]:
        output_index, content_index, item_state = self._message_state(event)
        part = event.get("part")
        if not isinstance(part, Mapping) or part.get("type") != "output_text":
            _raise_malformed("QwenCloud stream content part is unsupported.")
        text = part.get("text", "")
        if not isinstance(text, str) or text:
            _raise_malformed("QwenCloud stream content part start is malformed.")
        key = (output_index, content_index)
        if key in self._text_parts:
            _raise_malformed("QwenCloud stream content index was reused.")
        self._text_parts[key] = _TextPartState(item_id=item_state.item_id)
        return ()

    def _text_state(
        self, event: Mapping[str, Any]
    ) -> tuple[_TextPartState, tuple[int, int]]:
        output_index, content_index, item_state = self._message_state(event)
        key = (output_index, content_index)
        state = self._text_parts.get(key)
        if state is None or state.item_id != item_state.item_id:
            _raise_malformed("QwenCloud stream text part was not established.")
        return state, key

    @staticmethod
    def _text_chunk(text: str) -> dict[str, Any]:
        return {"choices": [{"delta": {"content": text}}]}

    def _handle_text_delta(
        self, event: Mapping[str, Any]
    ) -> tuple[dict[str, Any], ...]:
        state, _ = self._text_state(event)
        if state.final_text is not None:
            _raise_malformed(
                "QwenCloud stream text delta arrived after text completion."
            )
        delta = _required_string(event, "delta", allow_empty=True)
        self._reserve_output(len(delta))
        state.fragments.append(delta)
        state.delta_count += 1
        return (self._text_chunk(delta),)

    def _accept_final_text(
        self, state: _TextPartState, text: Any
    ) -> tuple[dict[str, Any], ...]:
        if not isinstance(text, str):
            _raise_malformed("QwenCloud stream final text is malformed.")
        if state.final_text is not None:
            if state.final_text != text:
                _raise_malformed("QwenCloud stream final text conflicts.")
            return ()
        if state.delta_count:
            emitted_text = "".join(state.fragments)
            if emitted_text != text:
                _raise_malformed("QwenCloud stream final text conflicts with deltas.")
            state.final_text = text
            state.fragments.clear()
            return ()
        self._reserve_output(len(text))
        state.final_text = text
        return (self._text_chunk(text),) if text else ()

    @staticmethod
    def _tool_chunk(
        state: _FunctionCallState,
        *,
        arguments: str,
        include_identity: bool = False,
    ) -> dict[str, Any]:
        function: dict[str, Any] = {"arguments": arguments}
        tool_call: dict[str, Any] = {
            "index": state.tool_index,
            "function": function,
        }
        if include_identity:
            tool_call.update({"id": state.call_id, "type": "function"})
            function["name"] = state.name
        return {"choices": [{"delta": {"tool_calls": [tool_call]}}]}

    def _call_state(self, event: Mapping[str, Any]) -> _FunctionCallState:
        output_index = _required_index(event, "output_index")
        item_id = _required_string(event, "item_id")
        item_state = self._output_items.get(output_index)
        call_state = self._function_calls.get(output_index)
        if (
            item_state is None
            or item_state.item_type != "function_call"
            or item_state.item_id != item_id
            or call_state is None
            or call_state.item_id != item_id
        ):
            _raise_malformed("QwenCloud stream function-call identity is inconsistent.")
        return call_state

    def _handle_arguments_delta(
        self, event: Mapping[str, Any]
    ) -> tuple[dict[str, Any], ...]:
        state = self._call_state(event)
        if state.final_arguments is not None:
            _raise_malformed(
                "QwenCloud stream function arguments arrived after completion."
            )
        delta = _required_string(event, "delta", allow_empty=True)
        self._reserve_output(len(delta))
        state.argument_fragments.append(delta)
        return (self._tool_chunk(state, arguments=delta),)

    @staticmethod
    def _validate_arguments_object(arguments: str) -> None:
        decoded = _strict_json_loads(arguments)
        if decoded is _JSON_DECODE_FAILED:
            _raise_malformed("QwenCloud stream function arguments are malformed.")
        if not isinstance(decoded, Mapping):
            _raise_malformed("QwenCloud stream function arguments must be an object.")

    def _accept_final_arguments(
        self, state: _FunctionCallState, arguments: Any
    ) -> tuple[dict[str, Any], ...]:
        if not isinstance(arguments, str):
            _raise_malformed("QwenCloud stream final function arguments are malformed.")
        if state.final_arguments is not None:
            if state.final_arguments != arguments:
                _raise_malformed("QwenCloud stream final function arguments conflict.")
            return ()
        emitted_arguments = "".join(state.argument_fragments)
        if not arguments.startswith(emitted_arguments):
            _raise_malformed(
                "QwenCloud stream final function arguments conflict with deltas."
            )
        self._validate_arguments_object(arguments)
        suffix = arguments[len(emitted_arguments) :]
        self._reserve_output(len(suffix))
        state.argument_fragments.clear()
        state.final_arguments = arguments
        return (self._tool_chunk(state, arguments=suffix),) if suffix else ()

    def _reserve_output(self, characters: int) -> None:
        if characters < 0 or self._output_chars + characters > _MAX_OUTPUT_CHARS:
            _raise_malformed("QwenCloud stream output limit was exceeded.")
        self._output_chars += characters

    def _handle_arguments_done(
        self, event: Mapping[str, Any]
    ) -> tuple[dict[str, Any], ...]:
        state = self._call_state(event)
        return self._accept_final_arguments(state, event.get("arguments"))

    def _handle_text_done(self, event: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
        state, _ = self._text_state(event)
        return self._accept_final_text(state, event.get("text"))

    def _handle_content_part_done(
        self, event: Mapping[str, Any]
    ) -> tuple[dict[str, Any], ...]:
        state, _ = self._text_state(event)
        part = event.get("part")
        if not isinstance(part, Mapping) or part.get("type") != "output_text":
            _raise_malformed("QwenCloud stream completed content part is malformed.")
        return self._accept_final_text(state, part.get("text"))

    def _validate_message_item(
        self,
        output_index: int,
        item: Mapping[str, Any],
        *,
        allowed_statuses: frozenset[str],
        mark_done: bool = False,
    ) -> tuple[dict[str, Any], ...]:
        item_state = self._output_items.get(output_index)
        if item_state is None or item_state.item_type != "message":
            _raise_malformed("QwenCloud stream message output was not established.")
        if item.get("type") != "message" or item.get("id") != item_state.item_id:
            _raise_malformed("QwenCloud stream message output identity conflicts.")
        self._validate_item_status(
            item_state,
            item,
            allowed_statuses=allowed_statuses,
            mark_done=mark_done,
        )
        content = item.get("content")
        if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
            _raise_malformed("QwenCloud stream message content is malformed.")

        chunks: list[dict[str, Any]] = []
        expected_keys: set[tuple[int, int]] = set()
        for content_index, part in enumerate(content):
            if not isinstance(part, Mapping) or part.get("type") != "output_text":
                _raise_malformed("QwenCloud stream message content is unsupported.")
            key = (output_index, content_index)
            expected_keys.add(key)
            state = self._text_parts.get(key)
            if state is None:
                state = _TextPartState(item_id=item_state.item_id)
                self._text_parts[key] = state
            elif state.item_id != item_state.item_id:
                _raise_malformed("QwenCloud stream text identity conflicts.")
            chunks.extend(self._accept_final_text(state, part.get("text")))
        if any(
            key[0] == output_index and key not in expected_keys
            for key in self._text_parts
        ):
            _raise_malformed("QwenCloud stream terminal text parts are incomplete.")
        if mark_done:
            item_state.done_status = cast(str, item.get("status"))
        return tuple(chunks)

    @staticmethod
    def _validate_item_status(
        item_state: _OutputItemState,
        item: Mapping[str, Any],
        *,
        allowed_statuses: frozenset[str],
        mark_done: bool,
    ) -> None:
        status = item.get("status")
        if not isinstance(status, str) or status not in allowed_statuses:
            _raise_malformed("QwenCloud stream output item status is malformed.")
        if item_state.done_status is not None:
            if mark_done:
                _raise_malformed(
                    "QwenCloud stream output item completed more than once."
                )
            if status != item_state.done_status:
                _raise_malformed("QwenCloud stream output item status conflicts.")

    def _handle_output_item_done(
        self, event: Mapping[str, Any]
    ) -> tuple[dict[str, Any], ...]:
        output_index = _required_index(event, "output_index")
        item = event.get("item")
        if not isinstance(item, Mapping):
            _raise_malformed("QwenCloud stream completed output item is malformed.")
        if item.get("type") == "reasoning":
            return self._validate_reasoning_item(
                output_index,
                item,
                allowed_statuses=frozenset({"completed", "incomplete"}),
                mark_done=True,
            )
        if item.get("type") == "function_call":
            return self._validate_function_item(output_index, item, mark_done=True)
        return self._validate_message_item(
            output_index,
            item,
            allowed_statuses=frozenset({"completed", "incomplete"}),
            mark_done=True,
        )

    def _validate_reasoning_item(
        self,
        output_index: int,
        item: Mapping[str, Any],
        *,
        allowed_statuses: frozenset[str],
        mark_done: bool = False,
    ) -> tuple[dict[str, Any], ...]:
        item_state = self._output_items.get(output_index)
        if (
            item_state is None
            or item_state.item_type != "reasoning"
            or item.get("type") != "reasoning"
            or item.get("id") != item_state.item_id
        ):
            _raise_malformed("QwenCloud stream reasoning identity conflicts.")
        self._validate_item_status(
            item_state,
            item,
            allowed_statuses=allowed_statuses,
            mark_done=mark_done,
        )
        if mark_done:
            item_state.done_status = cast(str, item.get("status"))
        return ()

    def _validate_function_item(
        self,
        output_index: int,
        item: Mapping[str, Any],
        *,
        mark_done: bool = False,
    ) -> tuple[dict[str, Any], ...]:
        item_state = self._output_items.get(output_index)
        call_state = self._function_calls.get(output_index)
        if (
            item_state is None
            or item_state.item_type != "function_call"
            or call_state is None
            or item.get("type") != "function_call"
            or item.get("id") != item_state.item_id
            or item.get("call_id") != call_state.call_id
            or item.get("name") != call_state.name
        ):
            _raise_malformed("QwenCloud stream completed function identity conflicts.")
        self._validate_item_status(
            item_state,
            item,
            allowed_statuses=frozenset({"completed"}),
            mark_done=mark_done,
        )
        chunks = self._accept_final_arguments(call_state, item.get("arguments"))
        if mark_done:
            item_state.done_status = cast(str, item.get("status"))
        return chunks

    def _handle_terminal(
        self, event_type: str, event: Mapping[str, Any]
    ) -> tuple[dict[str, Any], ...]:
        response = event.get("response")
        if not isinstance(response, Mapping):
            _raise_malformed("QwenCloud stream terminal response is malformed.")
        status = response.get("status")
        expected_status = event_type.removeprefix("response.")
        if status != expected_status:
            _raise_malformed("QwenCloud stream terminal status is malformed.")
        if status in {"failed", "cancelled"}:
            raise _stream_error("QwenCloud did not complete the streamed response.")
        if status not in {"completed", "incomplete"}:
            _raise_malformed("QwenCloud stream terminal status is unsupported.")

        output = response.get("output")
        if not isinstance(output, Sequence) or isinstance(output, (str, bytes)):
            _raise_malformed("QwenCloud stream terminal output is malformed.")
        chunks: list[dict[str, Any]] = []
        terminal_indexes: set[int] = set()
        for output_index, item in enumerate(output):
            if not isinstance(item, Mapping):
                _raise_malformed("QwenCloud stream terminal output item is malformed.")
            terminal_indexes.add(output_index)
            item_type = item.get("type")
            if item_type == "reasoning":
                reasoning_statuses = (
                    frozenset({"completed"})
                    if status == "completed"
                    else frozenset({"completed", "incomplete"})
                )
                chunks.extend(
                    self._validate_reasoning_item(
                        output_index,
                        item,
                        allowed_statuses=reasoning_statuses,
                    )
                )
                continue
            if item_type == "function_call":
                chunks.extend(self._validate_function_item(output_index, item))
                continue
            if item_type != "message":
                _raise_malformed("QwenCloud stream terminal output is unsupported.")
            chunks.extend(
                self._validate_message_item(
                    output_index,
                    item,
                    allowed_statuses=frozenset({status}),
                )
            )
        if set(self._output_items) != terminal_indexes:
            _raise_malformed("QwenCloud stream terminal output is incomplete.")

        usable_text = any(
            state.final_text is not None and state.final_text.strip()
            for state in self._text_parts.values()
        )
        complete_calls = bool(self._function_calls) and all(
            state.final_arguments is not None for state in self._function_calls.values()
        )
        if self._function_calls and not complete_calls:
            _raise_malformed("QwenCloud stream function call is incomplete.")
        if not usable_text and not complete_calls:
            _raise_malformed("QwenCloud stream returned no usable content.")
        if status == "incomplete":
            details = response.get("incomplete_details")
            reason = details.get("reason") if isinstance(details, Mapping) else None
            if reason != "max_output_tokens" or complete_calls:
                _raise_malformed("QwenCloud stream incomplete reason is unsupported.")
            finish_reason = "length"
        elif complete_calls:
            finish_reason = "tool_calls"
        else:
            finish_reason = "stop"

        usage = response.get("usage", {})
        if not isinstance(usage, Mapping):
            _raise_malformed("QwenCloud stream terminal usage is malformed.")
        chunks.append(
            {
                "choices": [{"delta": {"content": ""}, "finish_reason": finish_reason}],
                "usage": deepcopy(dict(usage)),
            }
        )
        self._terminal = True
        return tuple(chunks)


class _ChatCompletionsStreamTranslator:
    """Validate Chat stream metadata without accumulating tool arguments."""

    _TERMINAL_REASONS = frozenset({"stop", "length", "tool_calls"})

    def __init__(self) -> None:
        self._choices: dict[int, _ChatChoiceState] = {}
        self._output_chars = 0
        self._usage_seen = False

    def feed(self, event: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
        choices = event.get("choices")
        usage = event.get("usage")
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
            _raise_malformed("QwenCloud chat stream choices are malformed.")
        if usage is not None and not isinstance(usage, Mapping):
            _raise_malformed("QwenCloud chat stream usage is malformed.")
        if self._usage_seen:
            _raise_malformed("QwenCloud chat stream data arrived after usage.")
        if not choices:
            if usage is None:
                _raise_malformed("QwenCloud chat stream event is empty.")
            if not self._choices or not self._all_choices_terminal():
                _raise_malformed("QwenCloud chat stream usage arrived before terminal.")
            self._usage_seen = True
            return (self._safe_event(event),)
        if usage is not None:
            _raise_malformed("QwenCloud chat stream usage event is malformed.")

        event_choice_indexes: set[int] = set()
        for choice in choices:
            if not isinstance(choice, Mapping):
                _raise_malformed("QwenCloud chat stream choice is malformed.")
            choice_index = _required_index(choice, "index")
            if choice_index in event_choice_indexes:
                _raise_malformed("QwenCloud chat stream choice index was duplicated.")
            event_choice_indexes.add(choice_index)
            state = self._choices.setdefault(
                choice_index, _ChatChoiceState(tools={}, call_ids=set())
            )
            if state.terminal_reason is not None:
                _raise_malformed("QwenCloud chat stream choice arrived after terminal.")
            delta = choice.get("delta")
            if not isinstance(delta, Mapping):
                _raise_malformed("QwenCloud chat stream delta is malformed.")
            self._validate_delta(state, delta)

            finish_reason = choice.get("finish_reason")
            if finish_reason is not None:
                if (
                    not isinstance(finish_reason, str)
                    or finish_reason not in self._TERMINAL_REASONS
                ):
                    _raise_malformed("QwenCloud chat stream finish state is malformed.")
                has_tools = bool(state.tools)
                if (finish_reason == "tool_calls") != has_tools:
                    _raise_malformed(
                        "QwenCloud chat stream finish state conflicts with tools."
                    )
                state.terminal_reason = finish_reason
        return (self._safe_event(event),)

    def finish(self) -> tuple[dict[str, Any], ...]:
        if not self._choices or not self._all_choices_terminal():
            _raise_malformed("QwenCloud chat stream terminal event is missing.")
        return ()

    def _all_choices_terminal(self) -> bool:
        return all(
            state.terminal_reason is not None for state in self._choices.values()
        )

    def _validate_delta(
        self, state: _ChatChoiceState, delta: Mapping[str, Any]
    ) -> None:
        if "role" in delta and delta.get("role") != "assistant":
            _raise_malformed("QwenCloud chat stream role is malformed.")
        if "content" in delta:
            content = delta.get("content")
            if content is not None and not isinstance(content, str):
                _raise_malformed("QwenCloud chat stream content is malformed.")
            if isinstance(content, str):
                self._reserve_output(len(content))
        tool_calls = delta.get("tool_calls")
        if tool_calls is None:
            return
        if not isinstance(tool_calls, Sequence) or isinstance(tool_calls, (str, bytes)):
            _raise_malformed("QwenCloud chat stream tool delta is malformed.")
        event_tool_indexes: set[int] = set()
        for tool_call in tool_calls:
            if not isinstance(tool_call, Mapping):
                _raise_malformed("QwenCloud chat stream tool call is malformed.")
            tool_index = _required_index(tool_call, "index")
            if tool_index in event_tool_indexes:
                _raise_malformed("QwenCloud chat stream tool index was duplicated.")
            event_tool_indexes.add(tool_index)
            function = tool_call.get("function")
            if not isinstance(function, Mapping):
                _raise_malformed("QwenCloud chat stream tool function is malformed.")
            argument_chars = 0
            if "arguments" in function:
                arguments = function.get("arguments")
                if not isinstance(arguments, str):
                    _raise_malformed(
                        "QwenCloud chat stream tool arguments are malformed."
                    )
                argument_chars = len(arguments)
            existing = state.tools.get(tool_index)
            if existing is None:
                call_id = _required_string(tool_call, "id")
                if tool_call.get("type") != "function":
                    _raise_malformed("QwenCloud chat stream tool type is malformed.")
                name = _required_string(function, "name")
                if call_id in state.call_ids:
                    _raise_malformed("QwenCloud chat stream tool ID was duplicated.")
                self._reserve_output(
                    _metadata_characters(call_id, name) + argument_chars
                )
                state.call_ids.add(call_id)
                state.tools[tool_index] = _ChatToolState(call_id=call_id, name=name)
                continue
            if "id" in tool_call and tool_call.get("id") != existing.call_id:
                _raise_malformed("QwenCloud chat stream tool identity conflicts.")
            if "type" in tool_call and tool_call.get("type") != "function":
                _raise_malformed("QwenCloud chat stream tool type conflicts.")
            if "name" in function and function.get("name") != existing.name:
                _raise_malformed("QwenCloud chat stream tool name conflicts.")
            self._reserve_output(argument_chars)

    def _reserve_output(self, characters: int) -> None:
        if characters < 0 or self._output_chars + characters > _MAX_OUTPUT_CHARS:
            _raise_malformed("QwenCloud chat stream output limit was exceeded.")
        self._output_chars += characters

    @staticmethod
    def _safe_event(event: Mapping[str, Any]) -> dict[str, Any]:
        """Copy only metadata consumed by the shared OpenAI accumulator."""
        result: dict[str, Any] = {}
        if "id" in event:
            result["id"] = _required_string(event, "id")
        choices = cast(Sequence[Mapping[str, Any]], event["choices"])
        safe_choices: list[dict[str, Any]] = []
        for choice in choices:
            safe_choice: dict[str, Any] = {"index": choice["index"]}
            delta = cast(Mapping[str, Any], choice["delta"])
            safe_delta: dict[str, Any] = {}
            if "role" in delta:
                safe_delta["role"] = delta["role"]
            if isinstance(delta.get("content"), str):
                safe_delta["content"] = delta["content"]
            if delta.get("tool_calls") is not None:
                safe_tools: list[dict[str, Any]] = []
                for tool in cast(Sequence[Mapping[str, Any]], delta["tool_calls"]):
                    safe_tool: dict[str, Any] = {"index": tool["index"]}
                    for key in ("id", "type"):
                        if key in tool:
                            safe_tool[key] = tool[key]
                    function = cast(Mapping[str, Any], tool["function"])
                    safe_function = {
                        key: function[key]
                        for key in ("name", "arguments")
                        if key in function
                    }
                    safe_tool["function"] = safe_function
                    safe_tools.append(safe_tool)
                safe_delta["tool_calls"] = safe_tools
            safe_choice["delta"] = safe_delta
            if "finish_reason" in choice:
                safe_choice["finish_reason"] = choice["finish_reason"]
            safe_choices.append(safe_choice)
        result["choices"] = safe_choices
        if isinstance(event.get("usage"), Mapping):
            result["usage"] = deepcopy(dict(cast(Mapping[str, Any], event["usage"])))
        return result


class QwenCloudStream(Iterator[dict[str, Any]]):
    """Own and normalize one live QwenCloud streaming response."""

    def __init__(
        self,
        *,
        response: requests.Response,
        session: requests.Session,
        api_mode: QwenCloudAPIMode,
    ) -> None:
        self._response = response
        self._session = session
        self._api_mode = api_mode
        self._records = iter_sse_data_records(response.iter_content(chunk_size=8192))
        self._translator = (
            QwenResponsesStreamTranslator()
            if api_mode == "responses"
            else _ChatCompletionsStreamTranslator()
        )
        self._pending: deque[dict[str, Any]] = deque()
        self._event_count = 0
        self._closed = False

    def __iter__(self) -> QwenCloudStream:
        return self

    def __next__(self) -> dict[str, Any]:
        if self._closed:
            raise StopIteration
        while not self._pending:
            record_result = self._read_record()
            if record_result is _STREAM_END:
                try:
                    finish_result = self._finish_translation()
                except ChatProviderError:
                    self.close()
                    raise
                if finish_result is _STREAM_TRANSLATION_FAILED:
                    self.close()
                    raise _stream_error("QwenCloud returned malformed streaming data.")
                self.close()
                if not self._pending:
                    raise StopIteration
                break
            if record_result is _STREAM_READ_FAILED:
                self.close()
                raise _stream_error("QwenCloud returned malformed streaming data.")
            record = cast(str, record_result)

            if record == "[DONE]":
                try:
                    finish_result = self._finish_translation()
                except ChatProviderError:
                    self.close()
                    raise
                if finish_result is _STREAM_TRANSLATION_FAILED:
                    self.close()
                    raise _stream_error("QwenCloud returned malformed streaming data.")
                self.close()
                if not self._pending:
                    raise StopIteration
                break

            try:
                translation_result = self._translate_record(record)
            except ChatProviderError:
                self.close()
                raise
            if translation_result is _STREAM_TRANSLATION_FAILED:
                self.close()
                raise _stream_error("QwenCloud returned malformed streaming data.")
        return self._pending.popleft()

    def _finish_translation(self) -> object:
        try:
            self._pending.extend(self._translator.finish())
        except ChatProviderError:
            raise
        except Exception:
            return _STREAM_TRANSLATION_FAILED
        return None

    def _translate_record(self, record: str) -> object:
        try:
            event = self._decode_event(record)
            if self._event_count >= _MAX_STREAM_EVENTS:
                _raise_malformed("QwenCloud stream event limit was exceeded.")
            self._event_count += 1
            self._pending.extend(self._translator.feed(event))
        except ChatProviderError:
            raise
        except Exception:
            return _STREAM_TRANSLATION_FAILED
        return None

    def _read_record(self) -> str | object:
        try:
            return next(self._records)
        except StopIteration:
            return _STREAM_END
        except Exception:
            return _STREAM_READ_FAILED

    def close(self) -> None:
        """Close the response and its dedicated session exactly once."""
        if self._closed:
            return
        self._closed = True
        _best_effort_close(self._response)
        _best_effort_close(self._session)

    @staticmethod
    def _decode_event(record: str) -> Mapping[str, Any]:
        decoded = _strict_json_loads(record)
        if decoded is _JSON_DECODE_FAILED:
            _raise_malformed("QwenCloud returned malformed streaming JSON.")
        if not isinstance(decoded, Mapping):
            _raise_malformed("QwenCloud streaming event must be an object.")
        if decoded.get("type") == "error" or "error" in decoded:
            raise _stream_error("QwenCloud reported a streaming provider error.")
        return decoded


def iter_sse_data_records(chunks: Iterable[bytes]) -> Iterator[str]:
    """Yield complete SSE data records from arbitrary response byte chunks.

    The parser incrementally decodes strict UTF-8, recognizes LF, CR, and CRLF,
    ignores comments and non-data fields, and joins multiple ``data:`` fields
    with newlines. A data record is emitted only after its blank terminator.

    Args:
        chunks: Raw response-body byte chunks.

    Yields:
        Complete data-field payloads without JSON decoding.

    Raises:
        UnicodeDecodeError: If the byte stream is not valid UTF-8.
        TypeError: If a chunk is not bytes.
        ValueError: If EOF interrupts a data record before its blank terminator.
    """
    decoder = SSERecordDecoder(
        max_bytes=None,
        max_line_chars=_MAX_SSE_LINE_CHARS,
        max_record_chars=_MAX_SSE_RECORD_CHARS,
        max_line_segments=_MAX_SSE_LINE_SEGMENTS,
        max_data_lines=_MAX_SSE_DATA_LINES,
    )
    for chunk in chunks:
        for record in decoder.feed(chunk):
            yield record.data
    for record in decoder.finish():
        yield record.data
