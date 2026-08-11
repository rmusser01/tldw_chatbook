"""Record-aware streaming normalization for the QwenCloud adapter."""

from __future__ import annotations

import codecs
import json
from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Never, cast

import requests

from tldw_chatbook.Chat.Chat_Deps import ChatProviderError

if TYPE_CHECKING:
    from tldw_chatbook.LLM_Calls.qwencloud import QwenCloudAPIMode

_JSON_DECODE_FAILED = object()
_STREAM_END = object()
_STREAM_READ_FAILED = object()


def _stream_error(message: str) -> ChatProviderError:
    return ChatProviderError(provider="qwencloud", message=message, status_code=502)


def _raise_malformed(message: str) -> Never:
    raise _stream_error(message)


def _reject_json_constant(_value: str) -> Never:
    raise ValueError


def _strict_json_loads(value: str) -> Any:
    try:
        return json.loads(value, parse_constant=_reject_json_constant)
    except (TypeError, ValueError):
        return _JSON_DECODE_FAILED


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


@dataclass
class _OutputItemState:
    item_id: str
    item_type: str


@dataclass
class _TextPartState:
    item_id: str
    emitted_text: str = ""
    delta_count: int = 0
    final_text: str | None = None


@dataclass
class _FunctionCallState:
    item_id: str
    call_id: str
    name: str
    tool_index: int
    emitted_arguments: str = ""
    final_arguments: str | None = None


class QwenResponsesStreamTranslator:
    """Translate stateful QwenCloud Responses events into chat deltas."""

    def __init__(self) -> None:
        self._seen_sequences: dict[int, dict[str, Any]] = {}
        self._highest_sequence = -1
        self._output_items: dict[int, _OutputItemState] = {}
        self._text_parts: dict[tuple[int, int], _TextPartState] = {}
        self._function_calls: dict[int, _FunctionCallState] = {}
        self._call_ids: set[str] = set()
        self._terminal = False

    def feed(self, event: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
        """Consume one decoded Responses event and return normalized chunks."""
        if self._terminal:
            _raise_malformed("QwenCloud stream event arrived after the terminal event.")
        if not isinstance(event, Mapping):
            _raise_malformed("QwenCloud stream event must be an object.")
        event_copy = deepcopy(dict(event))
        sequence = event.get("sequence_number")
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
            _raise_malformed("QwenCloud stream sequence number is malformed.")
        previous = self._seen_sequences.get(sequence)
        if previous is not None:
            if previous == event_copy:
                return ()
            _raise_malformed("QwenCloud stream sequence replay conflicts.")
        if sequence <= self._highest_sequence:
            _raise_malformed("QwenCloud stream sequence number decreased.")
        self._seen_sequences[sequence] = event_copy
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
        if output_index in self._output_items:
            _raise_malformed("QwenCloud stream output index was reused.")
        if any(state.item_id == item_id for state in self._output_items.values()):
            _raise_malformed("QwenCloud stream output item identity was reused.")
        if item_type not in {"message", "reasoning", "function_call"}:
            _raise_malformed("QwenCloud stream output item type is unsupported.")
        self._output_items[output_index] = _OutputItemState(item_id, item_type)
        if item_type == "function_call":
            call_id = _required_string(item, "call_id")
            name = _required_string(item, "name")
            arguments = _required_string(item, "arguments", allow_empty=True)
            if call_id in self._call_ids:
                _raise_malformed("QwenCloud stream function-call ID was reused.")
            self._call_ids.add(call_id)
            call_state = _FunctionCallState(
                item_id=item_id,
                call_id=call_id,
                name=name,
                tool_index=len(self._function_calls),
                emitted_arguments=arguments,
            )
            self._function_calls[output_index] = call_state
            return (
                self._tool_chunk(
                    call_state,
                    arguments=arguments,
                    include_identity=True,
                ),
            )
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
        state.emitted_text += delta
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
            if state.emitted_text != text:
                _raise_malformed("QwenCloud stream final text conflicts with deltas.")
            state.final_text = text
            return ()
        state.emitted_text = text
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
        state.emitted_arguments += delta
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
        if not arguments.startswith(state.emitted_arguments):
            _raise_malformed(
                "QwenCloud stream final function arguments conflict with deltas."
            )
        self._validate_arguments_object(arguments)
        suffix = arguments[len(state.emitted_arguments) :]
        state.emitted_arguments = arguments
        state.final_arguments = arguments
        return (self._tool_chunk(state, arguments=suffix),) if suffix else ()

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
    ) -> tuple[dict[str, Any], ...]:
        item_state = self._output_items.get(output_index)
        if item_state is None or item_state.item_type != "message":
            _raise_malformed("QwenCloud stream message output was not established.")
        if item.get("type") != "message" or item.get("id") != item_state.item_id:
            _raise_malformed("QwenCloud stream message output identity conflicts.")
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
        return tuple(chunks)

    def _handle_output_item_done(
        self, event: Mapping[str, Any]
    ) -> tuple[dict[str, Any], ...]:
        output_index = _required_index(event, "output_index")
        item = event.get("item")
        if not isinstance(item, Mapping):
            _raise_malformed("QwenCloud stream completed output item is malformed.")
        if item.get("type") == "reasoning":
            item_state = self._output_items.get(output_index)
            if (
                item_state is None
                or item_state.item_type != "reasoning"
                or item.get("id") != item_state.item_id
            ):
                _raise_malformed("QwenCloud stream reasoning identity conflicts.")
            return ()
        if item.get("type") == "function_call":
            return self._validate_function_item(output_index, item)
        return self._validate_message_item(output_index, item)

    def _validate_function_item(
        self, output_index: int, item: Mapping[str, Any]
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
        return self._accept_final_arguments(call_state, item.get("arguments"))

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
                item_state = self._output_items.get(output_index)
                if item_state is not None and (
                    item_state.item_type != "reasoning"
                    or item.get("id") != item_state.item_id
                ):
                    _raise_malformed("QwenCloud stream reasoning identity conflicts.")
                continue
            if item_type == "function_call":
                chunks.extend(self._validate_function_item(output_index, item))
                continue
            if item_type != "message":
                _raise_malformed("QwenCloud stream terminal output is unsupported.")
            chunks.extend(self._validate_message_item(output_index, item))
        if set(self._output_items) != terminal_indexes:
            _raise_malformed("QwenCloud stream terminal output is incomplete.")

        usable_text = any(
            state.emitted_text.strip() for state in self._text_parts.values()
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
            QwenResponsesStreamTranslator() if api_mode == "responses" else None
        )
        self._pending: deque[dict[str, Any]] = deque()
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
                    if self._translator is not None:
                        self._pending.extend(self._translator.finish())
                except ChatProviderError:
                    self.close()
                    raise
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
                    if self._translator is not None:
                        self._pending.extend(self._translator.finish())
                except ChatProviderError:
                    self.close()
                    raise
                self.close()
                if not self._pending:
                    raise StopIteration
                break

            try:
                event = self._decode_event(record)
                if self._translator is None:
                    self._pending.append(self._normalize_chat_event(event))
                else:
                    self._pending.extend(self._translator.feed(event))
            except ChatProviderError:
                self.close()
                raise
        return self._pending.popleft()

    def _read_record(self) -> str | object:
        try:
            return next(self._records)
        except StopIteration:
            return _STREAM_END
        except (TypeError, UnicodeDecodeError, ValueError):
            return _STREAM_READ_FAILED

    def close(self) -> None:
        """Close the response and its dedicated session exactly once."""
        if self._closed:
            return
        self._closed = True
        try:
            self._response.close()
        finally:
            self._session.close()

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

    @staticmethod
    def _normalize_chat_event(event: Mapping[str, Any]) -> dict[str, Any]:
        choices = event.get("choices")
        usage = event.get("usage")
        if choices is None:
            _raise_malformed("QwenCloud chat stream choices are missing.")
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
            _raise_malformed("QwenCloud chat stream choices are malformed.")
        if usage is not None and not isinstance(usage, Mapping):
            _raise_malformed("QwenCloud chat stream usage is malformed.")
        if not choices and usage is None:
            _raise_malformed("QwenCloud chat stream event is empty.")
        for choice in choices:
            if not isinstance(choice, Mapping):
                _raise_malformed("QwenCloud chat stream choice is malformed.")
            delta = choice.get("delta")
            if not isinstance(delta, Mapping):
                _raise_malformed("QwenCloud chat stream delta is malformed.")
            finish_reason = choice.get("finish_reason")
            if finish_reason is not None and not isinstance(finish_reason, str):
                _raise_malformed("QwenCloud chat stream finish state is malformed.")
            tool_calls = delta.get("tool_calls")
            if tool_calls is not None and (
                not isinstance(tool_calls, Sequence)
                or isinstance(tool_calls, (str, bytes))
            ):
                _raise_malformed("QwenCloud chat stream tool delta is malformed.")
        return deepcopy(dict(event))


def _iter_utf8_lines(chunks: Iterable[bytes]) -> Iterator[tuple[str, bool]]:
    """Decode arbitrary byte chunks and yield universal-newline-delimited lines."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    buffered = ""
    for chunk in chunks:
        if not isinstance(chunk, bytes):
            raise TypeError("QwenCloud SSE chunks must be bytes.")
        buffered += decoder.decode(chunk, final=False)
        cursor = 0
        while cursor < len(buffered):
            lf_index = buffered.find("\n", cursor)
            cr_index = buffered.find("\r", cursor)
            indexes = [index for index in (lf_index, cr_index) if index >= 0]
            if not indexes:
                break
            newline_index = min(indexes)
            newline = buffered[newline_index]
            if newline == "\r" and newline_index + 1 == len(buffered):
                break
            yield buffered[cursor:newline_index], True
            cursor = newline_index + 1
            if newline == "\r" and buffered[cursor : cursor + 1] == "\n":
                cursor += 1
        buffered = buffered[cursor:]

    buffered += decoder.decode(b"", final=True)
    cursor = 0
    while cursor < len(buffered):
        lf_index = buffered.find("\n", cursor)
        cr_index = buffered.find("\r", cursor)
        indexes = [index for index in (lf_index, cr_index) if index >= 0]
        if not indexes:
            break
        newline_index = min(indexes)
        newline = buffered[newline_index]
        yield buffered[cursor:newline_index], True
        cursor = newline_index + 1
        if newline == "\r" and buffered[cursor : cursor + 1] == "\n":
            cursor += 1
    if cursor < len(buffered):
        yield buffered[cursor:], False


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
    data_lines: list[str] = []
    for line, terminated in _iter_utf8_lines(chunks):
        if not terminated:
            if line.startswith("data:") or line == "data" or data_lines:
                raise ValueError("QwenCloud SSE data record is incomplete.")
            continue
        if line == "":
            if data_lines:
                yield "\n".join(data_lines)
                data_lines.clear()
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if field != "data":
            continue
        if not separator:
            value = ""
        elif value.startswith(" "):
            value = value[1:]
        data_lines.append(value)

    if data_lines:
        raise ValueError("QwenCloud SSE data record is incomplete.")
