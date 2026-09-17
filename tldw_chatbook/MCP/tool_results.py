"""Complete MCP results and host-only wire/dispatch observations.

Display mappings deliberately carry no protocol provenance. Hook consumers must
use ``encoded_payload`` (when available), never mutable display/model fields.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock
from types import SimpleNamespace
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

MAX_OUTPUT_LINE_BYTES = 1_048_576
MAX_RESULT_BYTES = 786_432
MAX_JSON_DEPTH = 64
DispatchState = Literal["not_started", "settled", "uncertain"]


@dataclass
class MCPDispatchObservation:
    """Per-call host observation; owns no timeout, authority or resource."""

    state: DispatchState = "not_started"
    _audit_claimed: bool = field(default=False, init=False, repr=False)
    _audit_lock: Any = field(
        default_factory=Lock, init=False, repr=False, compare=False
    )

    def claim_audit(self) -> bool:
        """Claim one best-effort publication attempt, without holding I/O locks."""
        with self._audit_lock:
            if self._audit_claimed:
                return False
            self._audit_claimed = True
            return True


_dispatch: ContextVar[MCPDispatchObservation | None] = ContextVar(
    "mcp_dispatch_observation", default=None
)


@contextmanager
def observe_dispatch(observation: MCPDispatchObservation) -> Iterator[None]:
    """Carry one existing dispatch owner's observation down its await chain."""
    token = _dispatch.set(observation)
    try:
        yield
    finally:
        _dispatch.reset(token)


def current_dispatch() -> MCPDispatchObservation:
    """Use the enclosing call's observation, or create one for a direct call."""
    return _dispatch.get() or MCPDispatchObservation()


def _encode_bounded(value: object, *, limit: int = MAX_RESULT_BYTES) -> bytes:
    def validate(item: object, depth: int, ancestors: set[int]) -> None:
        if depth > MAX_JSON_DEPTH:
            raise ValueError
        if isinstance(item, (dict, list, tuple)):
            if id(item) in ancestors:
                raise ValueError
            ancestors.add(id(item))
            try:
                if isinstance(item, dict):
                    if not all(isinstance(key, str) for key in item):
                        raise ValueError
                    children = item.values()
                else:
                    children = item
                for child in children:
                    validate(child, depth + 1, ancestors)
            finally:
                ancestors.remove(id(item))
        elif item is None or isinstance(item, (str, bool, int)):
            return
        elif not isinstance(item, float) or not math.isfinite(item):
            raise ValueError

    try:
        validate(value, 1, set())
        chunks = []
        size = 0
        for chunk in json.JSONEncoder(
            ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).iterencode(value):
            encoded = chunk.encode("utf-8")
            size += len(encoded)
            if size > limit:
                raise ValueError
            chunks.append(encoded)
        return b"".join(chunks)
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise ValueError("mcp_result_invalid") from None


class MCPToolResult(BaseModel):
    """Owned complete result fields, with private host provenance."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    content: tuple[dict[str, Any], ...] = ()
    structured_content: dict[str, Any] | None = None
    is_error: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    transport_error: str | None = None
    _encoded_payload: bytes | None = PrivateAttr(default=None)
    _field_snapshot: bytes | None = PrivateAttr(default=None)
    _dispatch_state: DispatchState = PrivateAttr(default="not_started")

    @property
    def encoded_payload(self) -> bytes | None:
        """Original result bytes, only while the public fields still agree."""
        if self._encoded_payload is None:
            return None
        try:
            if (
                hashlib.sha256(
                    _encode_bounded(self.model_dump(), limit=MAX_RESULT_BYTES + 256)
                ).digest()
                == self._field_snapshot
            ):
                return self._encoded_payload
        except ValueError:
            pass
        return None

    @property
    def duplicate_keys_checked(self) -> bool:
        """Whether unchanged fields still have strict original-wire evidence."""
        return self.encoded_payload is not None

    @property
    def dispatch_state(self) -> DispatchState:
        """Host dispatch snapshot; remote bodies cannot supply this value."""
        return self._dispatch_state

    def __copy__(self) -> Self:
        copied = super().__copy__()
        copied._encoded_payload = None
        copied._field_snapshot = None
        return copied

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self:
        copied = super().__deepcopy__(memo)
        copied._encoded_payload = None
        copied._field_snapshot = None
        return copied

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        if update and set(update) - self.__class__.model_fields.keys():
            raise ValueError("mcp_result_invalid")
        return super().model_copy(update=update, deep=deep)


class _WireResult(dict):
    """Raw-reader-owned result mapping plus exact value span."""

    __slots__ = ("_encoded",)

    def __init__(self, value: dict[str, Any], encoded: bytes):
        super().__init__(value)
        self._encoded = encoded

    @property
    def encoded(self) -> bytes:
        return self._encoded


def parse_tool_result(payload: object) -> MCPToolResult:
    """Validate/copy a raw or compatible model result without inventing evidence.

    Raises:
        ValueError: With a fixed diagnostic for invalid/overflowing results.
    """
    encoded = payload.encoded if isinstance(payload, _WireResult) else None
    dispatch_state = "not_started"
    if isinstance(payload, MCPToolResult):
        if payload.transport_error is not None:
            return payload.model_copy(deep=True)
        dispatch_state = payload.dispatch_state
        # A copied/mutated model cannot recover its original qualification.
        encoded = payload.encoded_payload
        payload = {
            "content": list(payload.content),
            "structuredContent": payload.structured_content,
            "isError": payload.is_error,
            "_meta": payload.metadata,
        }
        if payload["structuredContent"] is None:
            del payload["structuredContent"]
    elif isinstance(payload, BaseModel):
        payload = payload.model_dump(by_alias=True, exclude_unset=True)
    elif isinstance(payload, SimpleNamespace):
        payload = vars(payload)
    if not isinstance(payload, Mapping):
        # Use one parser failure contract for invalid shape and value.
        raise ValueError("mcp_result_invalid")  # noqa: TRY004
    flag = payload.get("isError", False)
    if type(flag) is not bool:
        raise ValueError("mcp_error_flag_invalid")
    owned = json.loads(_encode_bounded(dict(payload)))
    content = owned.get("content", [])
    structured = owned.get("structuredContent")
    metadata = owned.get("_meta", {})
    if (
        not isinstance(content, list)
        or not all(isinstance(block, dict) for block in content)
        or ("structuredContent" in owned and not isinstance(structured, dict))
        or not isinstance(metadata, dict)
    ):
        raise ValueError("mcp_result_invalid")
    result = MCPToolResult(
        content=tuple(content),
        structured_content=structured,
        is_error=flag,
        metadata=metadata,
    )
    result._dispatch_state = dispatch_state
    if encoded is not None:
        if len(encoded) > MAX_RESULT_BYTES:
            raise ValueError("mcp_result_invalid")
        # The transport wrapper may have been mutated before this parser runs.
        if _encode_bounded(json.loads(encoded)) == _encode_bounded(owned):
            result._encoded_payload = encoded
            result._field_snapshot = hashlib.sha256(
                _encode_bounded(result.model_dump(), limit=MAX_RESULT_BYTES + 256)
            ).digest()
    return result


def transport_failure(code: str, observation: MCPDispatchObservation) -> MCPToolResult:
    """Build a host diagnostic, never interpolate remote exception text."""
    result = MCPToolResult(transport_error=code)
    result._dispatch_state = observation.state
    return result


def project_tool_result(result: MCPToolResult) -> dict[str, Any]:
    """Explicit legacy display shape, detached from complete protocol data."""
    if result.transport_error:
        return {"error": result.transport_error}
    if result.is_error:
        return {"error": "mcp_tool_error"}
    return {"result": json.loads(_encode_bounded(result.content))}


def _unique_object(pairs):
    value = {}
    for key, child in pairs:
        if key in value:
            raise ValueError("mcp_protocol_invalid")
        value[key] = child
    return value


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("mcp_protocol_invalid")
    return number


def _invalid_constant(_value):
    raise ValueError("mcp_protocol_invalid")


def decode_protocol_frame(frame: bytes) -> object:
    """Strict bounded JSON-RPC decode retaining exact result spans, also in batches."""
    if len(frame) > MAX_OUTPUT_LINE_BYTES:
        raise ValueError("mcp_protocol_invalid")
    text = frame.decode("utf-8")
    # Check nesting before the JSON decoder recurses; braces in strings do not count.
    depth, quoted, escaped = 0, False, False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > MAX_JSON_DEPTH:
                raise ValueError("mcp_protocol_invalid")
        elif char in "]}":
            depth -= 1
    decoder = json.JSONDecoder(
        object_pairs_hook=_unique_object,
        parse_constant=_invalid_constant,
        parse_float=_finite_float,
    )
    value = decoder.decode(text)

    def skip_space(index):
        while index < len(text) and text[index] in " \t\r\n":
            index += 1
        return index

    def envelope(index, message):
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            raise ValueError("mcp_protocol_invalid")
        index = skip_space(index + 1)
        while text[index] != "}":
            key, index = decoder.raw_decode(text, index)
            index = skip_space(index)
            index = skip_space(index + 1)  # validated colon
            start = index
            field, index = decoder.raw_decode(text, index)
            if key == "result":
                if not isinstance(field, dict):
                    raise ValueError("mcp_protocol_invalid")
                message[key] = _WireResult(field, text[start:index].encode("utf-8"))
            index = skip_space(index)
            if text[index] == ",":
                index = skip_space(index + 1)
        return message

    start = skip_space(0)
    if isinstance(value, list):
        index = skip_space(start + 1)
        for message in value:
            envelope(index, message)
            _, index = decoder.raw_decode(text, index)
            index = skip_space(index)
            if text[index] == ",":
                index = skip_space(index + 1)
    else:
        envelope(start, value)
    return value
